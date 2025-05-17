import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from correlation import correlation_analysis
from questions import questions_list

def filter_features(df, features_list, keep_label=None):
  df = df.copy()
  features_to_keep = [f for f in features_list if f in df.columns]
  if keep_label and keep_label in df.columns:
    features_to_keep.append(keep_label)
  return df[features_to_keep]

def label_encode_columns(df, columns, shared_encoder=False):
  df_encoded = df.copy()

  if shared_encoder:
    all_values = pd.concat([df_encoded[col].astype(str) for col in columns], axis=0)
    le = LabelEncoder()
    le.fit(all_values)
    for col in columns:
      df_encoded[col] = le.transform(df_encoded[col].astype(str))
  else:
    for col in columns:
      le = LabelEncoder()
      df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

  return df_encoded

def one_hot_encode_columns(df, columns):
  df_encoded = df.copy()
  encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int)
  transformed = encoder.fit_transform(df_encoded[columns])

  encoded_df = pd.DataFrame(
    transformed,
    columns=encoder.get_feature_names_out(columns),
    index=df_encoded.index
  )

  df_encoded.drop(columns=columns, inplace=True)
  df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

  return df_encoded

def split_mo_codes(df):
  df = df.copy()
  splitted_mo_codes = df["mo_code"].apply(lambda x: str(x).split(" "))
  max_register_mo_codes = splitted_mo_codes.apply(len).max()

  for i in range(max_register_mo_codes):
    df[f"mo_code{i+1}"] = splitted_mo_codes.apply(lambda x: x[i] if i < len(x) else "0")

  df.drop(columns=["mo_code"], inplace=True)
  mo_code_cols = [f"mo_code{i}" for i in range(1, 11)]
  df["mo_code_count"] = df[mo_code_cols].apply(lambda row: sum(val != "0" for val in row), axis=1)

  return df

def replace_null_values(df, columns, default_value, target_type=None):
  df = df.copy()
  for col in columns:
    if col in df.columns:
      df[col] = df[col].fillna(default_value)
      if target_type is not None:
        df[col] = df[col].astype(target_type)
  return df

def pre_process_dataset(dataset_path, question_index):
  label = questions_list[question_index]["label"]
  df = pd.read_csv(dataset_path, sep=";")
  print(df)

  #if question_index == 2:
    #df = df[df["status_code"] != "IC"]
  df = split_mo_codes(df)
  df = replace_null_values(df, ["weapon_code", "crime_type_code2"], 0, int)

  # encoding nominal features
  df = one_hot_encode_columns(df, [
    "department_code", 
    "vict_sex", 
    "vict_descent_code",
    "occurence_date_weekday",
    "register_date_weekday"
  ])
  df = label_encode_columns(df, [
    "status_code",
    "structure_code",
    "weapon_code",
    "severity_code"
  ])
  df = label_encode_columns(df, [
    "crime_type_code",
    "crime_type_code2"
  ], shared_encoder=True)
  df = label_encode_columns(df, [
    f"mo_code{i}" for i in range(1, 11) 
  ], shared_encoder=True)

  # normalize the data
  # scaler = MinMaxScaler()
  # scaler.fit(df)
  # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

  correlated_features = correlation_analysis(df, question_index)
  df = filter_features(df, correlated_features, label)

  labels = df[label]
  df.drop(columns=[label], inplace=True)

  return (df, labels)
