import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from transformers import TrainerCallback

class RealTimeMonitor(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.fig, self.axes = None, None

    def on_init_end(self, args, state, control, **kwargs):
        """训练初始化结束时调用"""
        print("训练初始化完成，开始监控...")

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时调用"""
        print("训练开始！")
        self.train_losses = []
        self.eval_losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次日志记录时调用"""
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
                self.update_display()

            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.update_display()

    def on_step_end(self, args, state, control, **kwargs):
        """每个训练步骤结束时调用"""
        pass  # 可以在这里添加步进监控

    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时调用"""
        if self.train_losses:
            print(f"Epoch {state.epoch} 完成，最新损失: {self.train_losses[-1]:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时调用"""
        print("训练完成！")
        self.save_final_plot()

    def update_display(self):
        """更新显示"""
        clear_output(wait=True)

        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 4))

        # 清空图表
        for ax in self.axes:
            ax.clear()

        # 训练损失
        if self.train_losses:
            self.axes[0].plot(self.steps, self.train_losses, 'b-', linewidth=2, label='Training Loss')
            self.axes[0].set_title('Training Loss')
            self.axes[0].set_xlabel('Steps')
            self.axes[0].set_ylabel('Loss')
            self.axes[0].grid(True, alpha=0.3)
            self.axes[0].legend()

        # 验证损失
        if len(self.eval_losses) > 0:
            eval_steps = self.steps[::max(1, len(self.steps)//len(self.eval_losses))][:len(self.eval_losses)]
            self.axes[1].plot(eval_steps, self.eval_losses, 'r-', linewidth=2, label='Evaluation Loss')
            self.axes[1].set_title('Evaluation Loss')
            self.axes[1].set_xlabel('Steps')
            self.axes[1].set_ylabel('Loss')
            self.axes[1].grid(True, alpha=0.3)
            self.axes[1].legend()
        else:
            self.axes[1].text(0.5, 0.5, '等待验证数据...',
                             ha='center', va='center', transform=self.axes[1].transAxes)
            self.axes[1].set_title('Evaluation Loss')

        plt.tight_layout()
        plt.show()

        # 打印当前状态
        self.print_status()

    def print_status(self):
        """打印当前训练状态"""
        current_step = self.steps[-1] if self.steps else 0
        current_loss = self.train_losses[-1] if self.train_losses else 0
        current_eval = self.eval_losses[-1] if self.eval_losses else None

        print("="*50)
        print(f"当前训练步数: {current_step}")
        print(f"最新训练损失: {current_loss:.4f}")
        if current_eval is not None:
            print(f"最新验证损失: {current_eval:.4f}")
        print("="*50)

    def save_final_plot(self):
        """保存最终训练图表"""
        if self.fig and self.train_losses:
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(self.steps, self.train_losses, 'b-', linewidth=2)
            plt.title('Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)

            if self.eval_losses:
                plt.subplot(1, 2, 2)
                eval_steps = self.steps[::max(1, len(self.steps)//len(self.eval_losses))][:len(self.eval_losses)]
                plt.plot(eval_steps, self.eval_losses, 'r-', linewidth=2)
                plt.title('Evaluation Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('./training_final.png', dpi=150, bbox_inches='tight')
            print("训练图表已保存为 'training_final.png'")