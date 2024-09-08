import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

def preprocess_dataset(data_id, class_indices=[0, 1], class_labels=[0, 1],
                       ohe=[], ordinal_encoding=[],remove=[]):
  print(f'preprocessing data with id = {data_id}...')
  # Fetch dataset
  dataset = fetch_ucirepo(id=data_id)

  # Data (as pandas dataframes)
  X = dataset.data.features
  y = dataset.data.targets

  # Remove unnecessary columns
  X = X.drop(remove, axis=1)

  # Combine features and targets into a single dataframe
  df = pd.concat([X, y], axis=1)

  # Get the name of the last column (target)
  target_column = df.columns[-1]

  # Take only the specified classes
  classes = pd.unique(np.squeeze(y))
  df = df[df[target_column].isin([classes[index] for index in class_indices])]

  # Map class labels to -1 and +1
  class_mapping = {classes[index]: label for (index, label) in zip(class_indices, class_labels)}
  df[target_column] = df[target_column].map(class_mapping)

  # One-hot encoding for specified columns and delete the original columns
  for col in ohe:
    if col in df.columns:
      dummies = pd.get_dummies(df[col], prefix=col)
      df = pd.concat([df, dummies], axis=1)
      df.drop(col, axis=1, inplace=True)
    else:
      print('Warning: column ', col, ' not found for one hot encoding')

  # Ordinal encoding
  for col in ordinal_encoding:
      if col in df.columns:
          encoder = OrdinalEncoder()
          df[col] = encoder.fit_transform(df[[col]])
      else:
        print('Warning: column ', col, ' not found for ordinal encoding')

  # Put the target at the end
  df = df[[col for col in df.columns if col != target_column] + [target_column]]

  # Normalize the features
  scaler = MinMaxScaler()
  scaled_values = scaler.fit_transform(df.iloc[:, :-1])

  # Convert scaled values to float and assign back to DataFrame
  # This maneuver is to avoid incompatible dtype warning
  for i, column in enumerate(df.columns[:-1]):
      df[column] = scaled_values[:, i].astype(float)

  # Shuffle the data
  df = shuffle(df, random_state=42).reset_index(drop=True)
  df.to_csv(f'data/{data_id}.csv', index=False, header=False)

  return df

if __name__ == '__main__':
    thyroid_ohe = ['Thyroid Function', 'Physical Examination', 'Adenopathy', 
               'Pathology', 'Risk', 'T', 'N', 'Stage', 'Response']
    thyroid_oe = ['Gender', 'Smoking','Hx Smoking', 'Hx Radiothreapy', 'M', 'Focality']
    students_ohe = ["Marital Status", "Application mode", "Course",
                              "Daytime/evening attendance", "Nacionality",
                              "Mother's occupation", "Father's occupation"]

    preprocess_dataset(53) # Iris
    preprocess_dataset(109) # Wine
    preprocess_dataset(327) # Phishing
    preprocess_dataset(697, ohe=students_ohe) # students success
    preprocess_dataset(732, remove=['ID']) # DARWIN
    preprocess_dataset(763) # Mines
    preprocess_dataset(759, ohe=['Race']) # Glioma
    preprocess_dataset(890, ohe=['trt']) # AIDs
    preprocess_dataset(915, ohe=thyroid_ohe, ordinal_encoding=thyroid_oe) # thyroid

    print('done preprocessing and saved files to /data')


