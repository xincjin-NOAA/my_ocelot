import os
import random
import pandas as pd
import lightning.pytorch as pl


# -----------------------------
# Random resampling per epoch
# -----------------------------
class ResampleDataCallback(pl.Callback):
    """
    Resample independent TRAIN and VAL windows from their respective date ranges.
    Updates happen at epoch END so that PL's reload_dataloaders_every_n_epochs
    will pick them up at the start of the next epoch.
    """

    def __init__(
        self,
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        train_window_days: int = 14,
        val_window_days: int = 3,
        mode: str = "random",
    ):
        # Training date range
        self.train_start_date = pd.to_datetime(train_start_date)
        self.train_end_date = pd.to_datetime(train_end_date)
        self.train_window_days = pd.Timedelta(days=train_window_days)

        # Validation date range
        self.val_start_date = pd.to_datetime(val_start_date)
        self.val_end_date = pd.to_datetime(val_end_date)
        self.val_window_days = pd.Timedelta(days=val_window_days)

        self.mode = mode
        self._check_date_ranges()

    # --- helpers -------------------------------------------------------------

    def _check_date_ranges(self):
        if self.train_end_date < self.train_start_date:
            raise ValueError(
                f"[Random Sampler] Training date range error: "
                f"Start ({self.train_start_date.date()}) exceeds end ({self.train_end_date.date()})"
            )
        if self.val_end_date < self.val_start_date:
            raise ValueError(
                f"[Random Sampler] Validation date range error: "
                f"Start ({self.val_start_date.date()}) exceeds end ({self.val_end_date.date()})"
            )

    @staticmethod
    def _clip_end(new_end: pd.Timestamp, end_limit: pd.Timestamp, tag: str) -> pd.Timestamp:
        if new_end > end_limit:
            print(f"[Random Sampler - {tag}] end clipped to {end_limit.date()}")
            return end_limit
        return new_end

    @staticmethod
    def _rand_offset(total_days: int, window_days: int) -> int:
        max_offset = max(0, total_days - window_days)
        return random.randint(0, max_offset) if max_offset > 0 else 0

    def _update_datamodule(self, trainer, start_dt, end_dt, is_train: bool):
        dm = trainer.datamodule
        if is_train:
            print(f"[DM.set_train_window] -> {start_dt} .. {end_dt}")
            dm.set_train_window(start_dt, end_dt)
        else:
            print(f"[DM.set_val_window]   -> {start_dt} .. {end_dt}")
            dm.set_val_window(start_dt, end_dt)
        # PL will reload dataloaders at the next epoch boundary
        # if reload_dataloaders_every_n_epochs=1 is set on the Trainer.

    # --- epoch hooks ---------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module):
        rank = int(os.environ.get("RANK", "0"))
        print(f"\n[Rank {rank}] === TRAIN EPOCH {pl_module.current_epoch} START ===")

    def on_validation_epoch_start(self, trainer, pl_module):
        rank = int(os.environ.get("RANK", "0"))
        print(f"\n[Rank {rank}] === VAL EPOCH {pl_module.current_epoch} START ===")

    def on_train_epoch_end(self, trainer, pl_module):
        # Prepare NEXT epoch's TRAIN window
        total_train_days = (self.train_end_date - self.train_start_date).days
        offset = self._rand_offset(total_train_days, self.train_window_days.days)
        new_start = self.train_start_date + pd.Timedelta(days=offset)
        new_end = self._clip_end(new_start + self.train_window_days, self.train_end_date, "TRAIN")

        rank = int(os.environ.get("RANK", "0"))
        print(f"[Rank {rank}] [Random Sampler - TRAIN] Next -> {new_start.date()} .. {new_end.date()}")
        self._update_datamodule(trainer, new_start, new_end, is_train=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Prepare NEXT epoch's VAL window
        total_val_days = (self.val_end_date - self.val_start_date).days
        offset = self._rand_offset(total_val_days, self.val_window_days.days)
        new_start = self.val_start_date + pd.Timedelta(days=offset)
        new_end = self._clip_end(new_start + self.val_window_days, self.val_end_date, "VAL")

        rank = int(os.environ.get("RANK", "0"))
        print(f"[Rank {rank}] [Random Sampler - VAL]   Next -> {new_start.date()} .. {new_end.date()}")
        self._update_datamodule(trainer, new_start, new_end, is_train=False)


# -----------------------------------
# Sequential (sliding) train windows
# -----------------------------------
class SequentialDataCallback(pl.Callback):
    """
    Slide a fixed-size TRAIN window forward through [full_start_date, full_end_date].
    Does not touch validation windows.
    """

    def __init__(self, full_start_date, full_end_date, window_days: int = 7, stride_days: int = 1):
        self.full_start_date = pd.to_datetime(full_start_date)
        self.full_end_date = pd.to_datetime(full_end_date)
        self.window = pd.Timedelta(days=window_days)
        self.stride = pd.Timedelta(days=stride_days)
        self.current_start = self.full_start_date

        if self.full_end_date <= self.full_start_date:
            raise ValueError(
                f"[Sequential Sampler] range invalid: {self.full_start_date.date()} .. {self.full_end_date.date()}"
            )

    def _window_for_current(self):
        start = self.current_start
        end = start + self.window
        if end > self.full_end_date:
            end = self.full_end_date
        return start, end

    def on_train_epoch_start(self, trainer, pl_module):
        # Set the TRAIN window for THIS epoch
        start, end = self._window_for_current()
        rank = int(os.environ.get("RANK", "0"))
        print(f"\n[Rank {rank}] [Sequential Sampler - TRAIN] Using -> {start.date()} .. {end.date()}")
        trainer.datamodule.set_train_window(start, end)

    def on_train_epoch_end(self, trainer, pl_module):
        # Advance for NEXT epoch
        self.current_start += self.stride
        if self.current_start >= self.full_end_date:
            print("[Sequential Sampler] Reached end of range; looping back.")
            self.current_start = self.full_start_date
