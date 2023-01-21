import pandas as pd

from utils import *

csv_path = 'training.csv'

df = pd.read_csv(csv_path)

visual_df_metric(df, "loss")

visual_df_metric(df, "dice_coeff")

visual_df_metric(df, "round_dice_coeff")