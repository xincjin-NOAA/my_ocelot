import lightning.pytorch as pl
import pandas as pd
import random


class ResampleDataCallback(pl.Callback):
    """
    Callback to resample from separate train/val date ranges
    """
    def __init__(self):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        """Sample from training date range"""
        datamodule = trainer.datamodule

        total_train_days = (datamodule.train_end_date - datamodule.train_start_date).days
        max_offset = total_train_days - datamodule.train_window_days.days
        random_day_offset = random.randint(0, max_offset) if max_offset > 0 else 0

        new_start = datamodule.train_start_date + pd.Timedelta(days=random_day_offset)
        new_end = new_start + datamodule.train_window_days

        print(f"\n[Random Sampler - TRAIN] Window: {new_start.date()} to {new_end.date()}\n")
        datamodule.set_train_data(new_start, new_end)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Sample from validation date range"""
        datamodule = trainer.datamodule

        total_val_days = (datamodule.val_end_date - datamodule.val_start_date).days
        max_offset = total_val_days - datamodule.val_window_days.days
        random_day_offset = random.randint(0, max_offset) if max_offset > 0 else 0

        new_start = datamodule.val_start_date + pd.Timedelta(days=random_day_offset)
        new_end = new_start + datamodule.val_window_days

        print(f"\n[Random Sampler - VAL] Window: {new_start.date()} to {new_end.date()}\n")
        datamodule.set_val_data(new_start, new_end)


class SequentialDataCallback(pl.Callback):
    """
    Callback to process data sequentially, one window at a time.

    At the beginning of each training epoch, this callback advances the
    datamodule's time window to the next chronological segment of the
    full dataset.
    """

    def __init__(self, full_start_date, full_end_date, window_days=7):
        self.full_start_date = pd.to_datetime(full_start_date)
        self.full_end_date = pd.to_datetime(full_end_date)
        self.window_days = pd.Timedelta(days=window_days)
        self.current_start_date = self.full_start_date

    def on_train_epoch_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # If we've gone past the end date, loop back to the beginning
        if self.current_start_date >= self.full_end_date:
            print(
                "\n[Sequential Sampler] Reached end of dataset. Looping back to the start."
            )
            self.current_start_date = self.full_start_date

        # 1. Define the window for the current epoch
        new_start = self.current_start_date
        new_end = new_start + self.window_days

        # 2. Ensure the window doesn't exceed the full end date
        if new_end > self.full_end_date:
            new_end = self.full_end_date

        print(
            f"\n[Sequential Sampler] Preparing new window: {new_start.date()} to {new_end.date()}\n"
        )

        # 3. Update the datamodule's hyperparameters with the new time window
        datamodule = trainer.datamodule
        datamodule.hparams.start_date = new_start
        datamodule.hparams.end_date = new_end

        # 4. Re-run the setup logic for the new window
        datamodule.setup("fit")

        # 5. Advance the start date for the next epoch
        # This creates an overlapping, "sliding" window
        self.current_start_date += pd.Timedelta(days=1)
