# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""训练/验证回调管理(train11111new 比赛版,原样提取自 utils/callbacks.py)。"""

import threading


# 训练生命周期回调注册与分发器(train/val 中用于挂载日志、保存等钩子)
class Callbacks:
    """Handles all registered callbacks for YOLOv3 Hooks."""

    def __init__(self):
        """Initializes a Callbacks object to manage YOLOv3 training hooks with various event triggers."""
        self._callbacks = {
            "on_pretrain_routine_start": [],
            "on_pretrain_routine_end": [],
            "on_train_start": [],
            "on_train_epoch_start": [],
            "on_train_batch_start": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],  # fit = train + val
            "on_model_save": [],
            "on_train_end": [],
            "on_params_update": [],
            "teardown": [],
        }
        self.stop_training = False  # set True to interrupt training

    # 注册回调到指定 hook
    def register_action(self, hook, name="", callback=None):
        """Register a new action to a callback hook.

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    # 依次触发指定 hook 下注册的所有回调
    def run(self, hook, *args, thread=False, **kwargs):
        """Loop through the registered actions and fire all callbacks on main thread.

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv3
            thread: (boolean) Run callbacks in daemon thread
            kwargs: Keyword Arguments to receive from YOLOv3
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        for logger in self._callbacks[hook]:
            if thread:
                threading.Thread(target=logger["callback"], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger["callback"](*args, **kwargs)
