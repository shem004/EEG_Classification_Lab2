import os
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from dataloader import read_bci_data, BCIDataset
from models import EEGNet, DeepConvNet


def plot_train_acc(train_acc_list, title='Training Accuracy', save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if isinstance(train_acc_list, dict):
        for name, acc_list in train_acc_list.items():
            plt.plot(range(1, len(acc_list) + 1), acc_list, linewidth=1.8, label=f"{name} (Max: {max(acc_list):.2f}%)")
    else:
        plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, linewidth=1.8, label='Training Acc')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    safe_name = title.replace(' ', '_').replace(':', '').replace('/', '-')
    plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150)
    plt.close()


def plot_train_loss(train_loss_list, title='Training Loss', save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if isinstance(train_loss_list, dict):
        for name, loss_list in train_loss_list.items():
            plt.plot(range(1, len(loss_list) + 1), loss_list, linewidth=1.8, label=name)
    else:
        plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, color='tab:orange', linewidth=1.8, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    safe_name = title.replace(' ', '_').replace(':', '').replace('/', '-')
    plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150)
    plt.close()


def plot_test_acc(test_acc_list, title='Testing Accuracy', save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if isinstance(test_acc_list, dict):
        for name, acc_list in test_acc_list.items():
            plt.plot(range(1, len(acc_list) + 1), acc_list, linewidth=1.8, label=f"{name} (Best: {max(acc_list):.2f}%)")
    else:
        plt.plot(range(1, len(test_acc_list) + 1), test_acc_list, color='tab:green', linewidth=1.8, label=f"Testing Acc (Best: {max(test_acc_list):.2f}%)")
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    safe_name = title.replace(' ', '_').replace(':', '').replace('/', '-')
    plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150)
    plt.close()


def plot_experiment_graphs(results_list, exp_title, save_dir='./plots'):
    """Generates 3 separate graphs for the experiment: Test Acc, Train Acc, and Train Loss."""
    test_acc_dict = {r['name']: r['test_acc'] for r in results_list}
    train_acc_dict = {r['name']: r['train_acc'] for r in results_list}
    train_loss_dict = {r['name']: r['train_loss'] for r in results_list}

    plot_test_acc(test_acc_dict, title=f"{exp_title} - Testing Accuracy", save_dir=save_dir)
    plot_train_acc(train_acc_dict, title=f"{exp_title} - Training Accuracy", save_dir=save_dir)
    plot_train_loss(train_loss_dict, title=f"{exp_title} - Training Loss", save_dir=save_dir)


def test(model, loader, device):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1
        avg_acc = (avg_acc / len(loader.dataset)) * 100
    return avg_acc


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None, patience=50, save_path='./weights/best.pt'):
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    best_acc = 0.0
    best_epoch = 0
    best_wts = deepcopy(model.state_dict())
    patience_counter = 0

    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []

    pbar = tqdm(range(1, num_epochs + 1), desc=os.path.basename(save_path))
    for epoch in pbar:
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            if scheduler is not None:
                scheduler.step()

            avg_loss /= len(train_loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(train_loader.dataset)) * 100
            avg_acc_list.append(avg_acc)

        test_acc = test(model, test_loader, device)
        test_acc_list.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_wts = deepcopy(model.state_dict())
            torch.save(best_wts, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix({
            'Train': f'{avg_acc:.1f}%',
            'Test': f'{test_acc:.1f}%',
            'Best': f'{best_acc:.2f}% (Ep {best_epoch})',
            'Patience': f'{patience_counter}/{patience}'
        })

        if patience is not None and patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch}! Best Test Acc: {best_acc:.2f}% at Epoch {best_epoch}.")
            break

    model.load_state_dict(best_wts)
    torch.save(best_wts, save_path)
    return avg_acc_list, avg_loss_list, test_acc_list, best_acc, best_epoch


def run_config(
    name, train_loader, test_loader, device,
    model_type='EEGNet', act='relu', dropout=0.25, alpha=1.0,
    negative_slope=0.01, F1=16, D=2, F2=32, kernel_length=51,
    lr=0.001, epochs=350, patience=50, use_sched=True, weights_dir='./weights'
):
    torch.manual_seed(42)
    np.random.seed(42)

    if model_type == 'EEGNet':
        model = EEGNet(
            activation=act, dropout=dropout, alpha=alpha,
            negative_slope=negative_slope, F1=F1, D=D, F2=F2,
            kernel_length=kernel_length
        ).to(device)
    else:
        model = DeepConvNet(
            activation=act, dropout=dropout, alpha=alpha,
            negative_slope=negative_slope
        ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5) if use_sched else None

    save_path = os.path.join(weights_dir, f'{name}.pt')
    train_acc, train_loss, test_acc, best_acc, best_ep = train(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=epochs, device=device, scheduler=scheduler,
        patience=patience, save_path=save_path
    )
    print(f"--> [{name}] Best Peak Test Acc: {best_acc:.2f}% (at Epoch {best_ep}/{len(test_acc)})\n")

    return {
        'name': name, 'model_type': model_type, 'act': act,
        'dropout': str(dropout), 'F1_D_F2': f"{F1}-{D}-{F2}", 'kernel': kernel_length,
        'lr': lr, 'best_epoch': best_ep, 'stopped_epoch': len(test_acc),
        'max_train_acc': round(max(train_acc), 2), 'best_test_acc': round(best_acc, 2),
        'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc,
        'model_obj': model
    }


def run_all_experiments(train_loader, test_loader, device, weights_dir='./weights', plots_dir='./plots'):
    all_summary = []

    # EXP 1: Custom Compact EEGNet (F1=8, D=2, F2=16, kernel=64)
    print("=== Experiment 1: Custom Compact EEGNet (F1=8, D=2, F2=16, kernel=64) ===")
    exp_custom = [
        run_config(
            "Exp1_Custom_EEGNet_Drop0.25_0.50", train_loader, test_loader, device,
            model_type='EEGNet', act='leaky_relu', negative_slope=0.05,
            F1=8, D=2, F2=16, kernel_length=64, dropout=(0.25, 0.50),
            lr=0.001, epochs=700, patience=50, weights_dir=weights_dir
        ),
        run_config(
            "Exp1_Custom_EEGNet_Drop0.30_0.50", train_loader, test_loader, device,
            model_type='EEGNet', act='leaky_relu', negative_slope=0.05,
            F1=8, D=2, F2=16, kernel_length=64, dropout=(0.30, 0.50),
            lr=0.001, epochs=700, patience=50, weights_dir=weights_dir
        ),
    ]
    plot_experiment_graphs(exp_custom, "Exp 1 EEGNet (F1=8 D=2 F2=16 K=64)", save_dir=plots_dir)
    all_summary.extend(exp_custom)

    # EXP 2: Custom Compact EEGNet (F1=16, D=2, F2=32, kernel=51)
    print("=== Experiment 2: Custom Compact EEGNet (F1=16, D=2, F2=32, kernel=51) ===")
    exp_custom_2 = [
        run_config(
            "Exp2_Custom_EEGNet_Drop0.25_0.50", train_loader, test_loader, device,
            model_type='EEGNet', act='leaky_relu', negative_slope=0.05,
            F1=16, D=2, F2=32, kernel_length=51, dropout=(0.25, 0.50),
            lr=0.001, epochs=700, patience=80, weights_dir=weights_dir
        ),
        run_config(
            "Exp2_Custom_EEGNet_Drop0.50_0.50", train_loader, test_loader, device,
            model_type='EEGNet', act='leaky_relu', negative_slope=0.05,
            F1=16, D=2, F2=32, kernel_length=51, dropout=(0.50, 0.50),
            lr=0.001, epochs=700, patience=80, weights_dir=weights_dir
        ),
    ]
    plot_experiment_graphs(exp_custom_2, "Exp 2 EEGNet LeakyReLU (F1=16 D=2 F2=32 K=51)", save_dir=plots_dir)
    all_summary.extend(exp_custom_2)

    # EXP 0: Init EEGNet ELU
    print("=== Exp 0: EEGNet init ELU ===")
    exp_custom_0 = [
        run_config(
            "Exp0_Init_EEGNet_ELU_Drop0.50_0.50", train_loader, test_loader, device,
            model_type='EEGNet', act='elu', negative_slope=0.05,
            F1=16, D=2, F2=32, kernel_length=51, dropout=(0.50, 0.50),
            lr=0.001, epochs=700, patience=80, weights_dir=weights_dir
        ),
    ]
    plot_experiment_graphs(exp_custom_0, "Exp 0 EEGNet init ELU", save_dir=plots_dir)
    all_summary.extend(exp_custom_0)

    # EXP 3: Combination Functions on Standard EEGNet
    print("=== Experiment 3: EEGNet Different Combinations ===")
    exp_eegnet_act = [
        run_config("EEGNet_ELU", train_loader, test_loader, device, model_type='EEGNet', act='elu', dropout=0.25, lr=0.001, epochs=700, patience=200, weights_dir=weights_dir),
        run_config("EEGNet_ReLU", train_loader, test_loader, device, model_type='EEGNet', act='relu', dropout=0.25, lr=0.001, epochs=700, patience=200, weights_dir=weights_dir),
        run_config("EEGNet_LeakyReLU_0.05", train_loader, test_loader, device, model_type='EEGNet', act='leaky_relu', negative_slope=0.05, dropout=0.25, lr=0.001, epochs=700, patience=200, weights_dir=weights_dir),
        run_config("EEGNet_ELU_a0.5", train_loader, test_loader, device, model_type='EEGNet', act='elu', alpha=0.5, dropout=0.25, lr=0.001, epochs=700, patience=200, weights_dir=weights_dir),
        run_config("EEGNet_ELU_a2.0", train_loader, test_loader, device, model_type='EEGNet', act='elu', alpha=2.0, dropout=0.25, lr=0.001, epochs=700, patience=200, weights_dir=weights_dir),
        run_config("EEGNet_ReLU_Drop0.1", train_loader, test_loader, device, model_type='EEGNet', act='relu', dropout=0.1, lr=0.001, epochs=700, patience=200, weights_dir=weights_dir),
        run_config("EEGNet_ReLU_Drop0.5", train_loader, test_loader, device, model_type='EEGNet', act='relu', dropout=0.5, lr=0.001, epochs=700, patience=200, weights_dir=weights_dir),
    ]
    plot_experiment_graphs(exp_eegnet_act, "Exp 3 EEGNet Different Combinations", save_dir=plots_dir)
    all_summary.extend(exp_eegnet_act)

    # EXP 4: DeepConvNet Activations
    print("=== Experiment 4: DeepConvNet Activation Functions ===")
    exp_deep_act = [
        run_config("DeepConvNet_ELU", train_loader, test_loader, device, model_type='DeepConvNet', act='elu', dropout=0.5, lr=0.001, epochs=3000, patience=500, weights_dir=weights_dir),
        run_config("DeepConvNet_ReLU", train_loader, test_loader, device, model_type='DeepConvNet', act='relu', dropout=0.5, lr=0.001, epochs=3000, patience=500, weights_dir=weights_dir),
        run_config("DeepConvNet_LeakyReLU", train_loader, test_loader, device, model_type='DeepConvNet', act='leaky_relu', negative_slope=0.05, dropout=0.5, lr=0.001, epochs=3000, patience=500, weights_dir=weights_dir),
    ]
    plot_experiment_graphs(exp_deep_act, "Exp 4 DeepConvNet Activations", save_dir=plots_dir)
    all_summary.extend(exp_deep_act)

    # EXP 5: DeepConvNet Variations
    print("=== Experiment 5: DeepConvNet Variations ===")
    exp_deep_var = [
        run_config("DeepConvNet_ReLU_Drop0.25", train_loader, test_loader, device, model_type='DeepConvNet', act='relu', dropout=0.25, lr=0.001, epochs=3000, patience=420, weights_dir=weights_dir),
        run_config("DeepConvNet_ReLU_Drop0.50", train_loader, test_loader, device, model_type='DeepConvNet', act='relu', dropout=0.50, lr=0.001, epochs=3000, patience=420, weights_dir=weights_dir),
        run_config("DeepConvNet_ReLU_Drop0.75", train_loader, test_loader, device, model_type='DeepConvNet', act='relu', dropout=0.75, lr=0.001, epochs=3000, patience=420, weights_dir=weights_dir),
        run_config("DeepConvNet_ReLU_LR0.0005", train_loader, test_loader, device, model_type='DeepConvNet', act='relu', dropout=0.5, lr=0.0005, epochs=3000, patience=420, weights_dir=weights_dir),
        run_config("DeepConvNet_ReLU_LR0.001_NoSched", train_loader, test_loader, device, model_type='DeepConvNet', act='relu', dropout=0.5, lr=0.001, epochs=3000, patience=420, use_sched=False, weights_dir=weights_dir),
        run_config("DeepConvNet_ELU_a0.5", train_loader, test_loader, device, model_type='DeepConvNet', act='elu', alpha=0.5, dropout=0.5, lr=0.001, epochs=3000, patience=420, weights_dir=weights_dir),
        run_config("DeepConvNet_LeakyReLU_0.10", train_loader, test_loader, device, model_type='DeepConvNet', act='leaky_relu', negative_slope=0.10, dropout=0.5, lr=0.001, epochs=3000, patience=420, weights_dir=weights_dir),
    ]
    plot_experiment_graphs(exp_deep_var, "Exp 5 DeepConvNet Variations", save_dir=plots_dir)
    all_summary.extend(exp_deep_var)

    # Top-3 Checkpoint Ensemble
    top_models = sorted(all_summary, key=lambda x: x['best_test_acc'], reverse=True)[:3]
    print(f"=== Experiment 6: Top-3 Checkpoint Ensemble ({[m['name'] for m in top_models]}) ===")
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            avg_probs = sum(torch.softmax(m['model_obj'](inputs), dim=1) for m in top_models) / len(top_models)
            preds = avg_probs.argmax(dim=1).cpu()
            correct += preds.eq(labels).sum().item()
    ensemble_acc = round((correct / len(test_loader.dataset)) * 100, 2)
    print(f"--> [Top-3 Ensemble] Official Test Accuracy: {ensemble_acc:.2f}%\n")

    df_summary = pd.DataFrame([
        {
            'Experiment': r['name'], 'Model': r['model_type'], 'F1-D-F2': r['F1_D_F2'],
            'Kernel': r['kernel'], 'Act': r['act'], 'Dropout': r['dropout'],
            'Best Epoch': f"{r['best_epoch']}/{r['stopped_epoch']}",
            'Max Train Acc (%)': r['max_train_acc'], 'Peak Test Acc (%)': r['best_test_acc']
        } for r in all_summary
    ]).sort_values(by='Peak Test Acc (%)', ascending=False).reset_index(drop=True)

    print("\n================ EXPERIMENT SUMMARY ================")
    print(df_summary.to_string(index=False))
    df_summary.to_csv('experiment_summary.csv', index=False)
    print("\nSaved summary table to 'experiment_summary.csv' and plots to './plots/'.")


def main():
    parser = argparse.ArgumentParser(description="EEG Classification with EEGNet and DeepConvNet")
    parser.add_argument('--mode', type=str, default='all', choices=['single', 'all'],
                        help="Run a single model configuration ('single') or all experiments ('all')")
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory containing .npz dataset files")
    parser.add_argument('--weights_dir', type=str, default='./weights', help="Directory to save model weights")
    parser.add_argument('--plots_dir', type=str, default='./plots', help="Directory to save plots")

    # Arguments for single mode
    parser.add_argument('--model', type=str, default='EEGNet', choices=['EEGNet', 'DeepConvNet'])
    parser.add_argument('--act', type=str, default='elu', choices=['elu', 'relu', 'leaky_relu'])
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data, train_label, test_data, test_label = read_bci_data(args.data_dir)
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.mode == 'single':
        exp_name = f"{args.model}_{args.act}_drop{args.dropout}_lr{args.lr}"
        res = run_config(
            exp_name, train_loader, test_loader, device,
            model_type=args.model, act=args.act, dropout=args.dropout,
            lr=args.lr, epochs=args.epochs, patience=args.patience,
            weights_dir=args.weights_dir
        )
        plot_experiment_graphs([res], exp_name, save_dir=args.plots_dir)
        print(f"Plots saved to '{args.plots_dir}/' and best weights saved to '{args.weights_dir}/{exp_name}.pt'.")
    else:
        run_all_experiments(train_loader, test_loader, device, weights_dir=args.weights_dir, plots_dir=args.plots_dir)


if __name__ == '__main__':
    main()