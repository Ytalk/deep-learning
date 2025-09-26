import keras # Neural Network Library
from keras import layers # Layers to a neural network
from keras import optimizers # optimizers
import pandas as pd # Data Manipulation library
import numpy as np # Fast Numeric Computing library
import tensorflow as tf # Optimizers
import matplotlib.pyplot as plt # Plot library
from tensorflow import keras
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
USE_TUNER = True
# =============================================================================

### 1. Carregamento e Limpeza ###
data = pd.read_csv('dow_jones_index.data') # Loading dataset
data.info() # inspecting columns and data types from "data" dataframe

# fixes columns with '$' symbol and converts columns to numeric type (float)
price_cols = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
for col in price_cols:
    data[col] = pd.to_numeric( data[col].astype(str).str.replace('$', '', regex=False), errors='coerce' )

data.dropna(inplace=True)
# fills null values with 0 only in the percentage change columns
##data['percent_change_volume_over_last_wk'].fillna(0, inplace=True)
##data['previous_weeks_volume'].fillna(0, inplace=True)


### Engenharia de Features ###
data['date'] = pd.to_datetime(data['date']) # 'data' -> datetime
data['semana_do_ano'] = data['date'].dt.isocalendar().week.astype(int)
data['mes'] = data['date'].dt.month.astype(int)

# 'stock' -> dummies (One-Hot Encoding). in that pipeline: One-Hot Encoding > Label Encoding
stock_dummies = pd.get_dummies(data['stock'], prefix='stock')
processed_data = pd.concat([data, stock_dummies], axis=1)



### Memory ###
# ensures data is sorted by stock and date so shift() works correctly
processed_data.sort_values(by=['stock', 'date'], inplace=True)
# Lag Features: how was performance last week? for each stock, get the value from the previous line with shift(1)
processed_data['lag_1_percent_change_price'] = processed_data.groupby('stock')['percent_change_price'].shift(1)
# Rolling Features: what's the recent trend? 3-week moving average for price and volume
processed_data['rolling_3_week_mean_price_change'] = processed_data.groupby('stock')['percent_change_price'].rolling(window=3).mean().reset_index(level=0, drop=True)
# Volatility Feature: is the stock stable or volatile? 3-week moving standard deviation for the price
processed_data['rolling_3_week_volatility'] = processed_data.groupby('stock')['percent_change_price'].rolling(window=3).std().reset_index(level=0, drop=True)


# the first lines of each stock will have NaN values because of the shift and rolling
processed_data.dropna(inplace=True)
'''processed_data['lag_1_percent_change_price'].fillna(0, inplace=True)
# Para as features de rolling, usamos backfill dentro de cada grupo de ação
cols_to_fill = ['rolling_3_week_mean_price_change', 'rolling_3_week_volatility']
for col in cols_to_fill:
    # O .groupby('stock') garante que o bfill não "vaze" dados de uma ação para outra
    processed_data[col] = processed_data.groupby('stock')[col].bfill()
# Pode haver algum NaN restante se uma ação tiver menos de 3 semanas de dados (raro)
# Usamos um fillna(0) final para garantir que não sobre nenhum NaN
processed_data.fillna(0, inplace=True)'''

data.info()

### Data Division ###
# isolates first and second quarter data
data_q1 = processed_data[processed_data['quarter'] == 1].copy()
data_q2 = processed_data[processed_data['quarter'] == 2].copy()

train_parts = [data_q1] # start the training list with the entire Q1
test_parts = []

# iterate over each stock in Q2 and split its data in half
for stock_symbol in data_q2['stock'].unique():
    # gets data from a single stock
    stock_data = data_q2[data_q2['stock'] == stock_symbol]

    # calculates the split point (half weeks for that stock)
    n_weeks = len(stock_data)
    split_point = n_weeks // 2

    # add the first half to training and the second half to testing
    train_parts.append(stock_data.iloc[:split_point])
    test_parts.append(stock_data.iloc[split_point:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

# remove the original columns
train_df = train_df.drop(columns=['stock', 'date', 'quarter'])
test_df = test_df.drop(columns=['stock', 'date', 'quarter'])

print(f"Tamanho do novo conjunto de treino: {len(train_df)} amostras")
print(f"Tamanho do novo conjunto de teste: {len(test_df)} amostras")

# defines the input (X) and output (Y) columns
output_col = 'percent_change_next_weeks_price' # Extract the output column
input_cols = train_df.drop(columns=output_col).columns # gets all other columns

train_x = train_df[input_cols]
train_y_reg = train_df[output_col]
test_x = test_df[input_cols]
test_y_reg = test_df[output_col]


# class 1: price went up (return > 0)
# class 0: price went down or stayed the same (return <= 0)
y_train_class = (train_y_reg > 0).astype(int)
y_test_class = (test_y_reg > 0).astype(int)

# normalizes data using Z-Score (apenas dados de treino)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)

#scaler_y = StandardScaler() # it is crucial to use a separate scaler for the Y
#y_train_scaled = scaler_y.fit_transform(train_y.values.reshape(-1, 1)) # .values.reshape(-1, 1) = array 2D

# verifying dataset dimensions
print('The training dataset (inputs) dimensions are: ', train_x.shape)
print('The training dataset (outputs) dimensions are: ', train_y_reg.shape)
print('The testing dataset (inputs) dimensions are: ', test_x.shape)
print('The testing dataset (outputs) dimensions are: ', test_y_reg.shape)


# Function to define model architecture
# (USE_TUNER = False)
def build_model_classification(input_shape):
    model = keras.Sequential([
        # input_shape is the amount of columns from training set
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') #output: 1 neuron with 'sigmoid' activation for probabilities
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# (USE_TUNER = True)
def build_model_classification_tuned(hp):
    model = keras.Sequential()

    model.add(layers.Input(shape=(len(input_cols),)))

    hp_units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32)
    model.add(layers.Dense(units=hp_units_1, activation='relu'))

    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    model.add(layers.Dropout(hp_dropout))

    hp_units_2 = hp.Int('units_2', min_value=32, max_value=256, step=32)
    model.add(layers.Dense(units=hp_units_2, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


if USE_TUNER:
    print("--- MODO: Otimização com Keras Tuner ---")
    # configures the hyperparameter "searcher"
    tuner = kt.RandomSearch(
        build_model_classification_tuned,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=15, # number of different combinations to test
        executions_per_trial=3, # how many times to train each combination
        directory='my_dir_clf',
        project_name='stock_classification_v2'
    )

    tuner.search(X_train_scaled, y_train_class, epochs=100, validation_split=0.2)


    print("\n--- Extraindo resultados de todos os trials do Tuner ---")

    all_trials = tuner.oracle.get_best_trials(num_trials=50) # 50 best trials

    results_list = []

    for trial in all_trials:
        # for each trial, extract the score (ex: val_accuracy) and the hyperparameters
        trial_score = trial.score
        trial_hps = trial.hyperparameters.values

        # adds the score to the hyperparameter dictionary
        trial_hps['score'] = trial_score

        results_list.append(trial_hps)

    # Pandas DataFrame with all results
    results_df = pd.DataFrame(results_list)

    # sort the DataFrame by score (from best to worst)
    results_df.sort_values(by='score', ascending=False, inplace=True)

    print("\n--- Top 15 Melhores Combinações Encontradas ---")
    print(results_df.head(15).to_string())

    results_df.to_csv('tuner_classification_results.csv', index=False)
    print("\nResultados completos salvos em 'tuner_classification_results.csv'")


    # pick the BEST HYPERPARAMETERS, not the model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    A busca terminou. Os melhores hiperparâmetros são:
    - Unidades na camada 1: {best_hps.get('units_1')}
    - Taxa de Dropout: {best_hps.get('dropout')}
    - Unidades na camada 2: {best_hps.get('units_2')}
    - Taxa de Aprendizado: {best_hps.get('learning_rate')}
    """)

    model = tuner.hypermodel.build(best_hps) # build the model
else:
    print("--- MODO: Treinamento com Modelo Fixo ---")
    model = build_model_classification([len(input_cols)])

model.summary()#table -> layer / shape / param

EPOCHS = 500
history = model.fit(
    X_train_scaled,
    y_train_class,
    epochs=EPOCHS,
    validation_split=0.2, # uses 20% of the training data for validation
    verbose=1
)

#line graph (MSE x Epoch)
plt.plot(history.history['loss'])
plt.title('Training Loss (MSE)')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(['Train Loss'], loc='upper right')
plt.show()


# create a boolean mask with obj pandas: true for columns that do NOT start with 'stock_' ('~' = NOT)
feature_names = pd.Series(input_cols)
mask = ~feature_names.str.startswith('stock_')

all_weights = model.get_weights()
weights_first_layer = all_weights[0]
#plot the weights of the first neuron of the first layer
plt.barh(feature_names[mask], weights_first_layer[mask, 0], align='center')
plt.xlabel("Weights")
plt.ylabel("Inputs")
plt.title("Weights of First Neuron in First Hidden Layer")
#plt.savefig("NN-Weights.png")
plt.show()



### Avaliação do Modelo ###
# predicts probabilities (values between 0 and 1)
probabilities = model.predict(X_test_scaled)
# converts probabilities to classes (0 or 1) using a threshold of 0.5
predictions = (probabilities > 0.5).astype(int)

# f1-score for each class
print(classification_report(y_test_class, predictions, target_names=['Desceu/Manteve', 'Subiu']))

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test_class, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Desceu/Manteve', 'Subiu'], yticklabels=['Desceu/Manteve', 'Subiu'])
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()



# predict radon activities with the built linear regression model
test_predictions_scaled = model.predict(X_test_scaled).flatten()

# scaler_y to reverse the transformation
#test_predictions = scaler_y.inverse_transform(test_predictions_scaled).flatten()

# results
plt.scatter(test_y_reg, test_predictions_scaled, marker='o', c='blue')
plt.plot([-5, 35], [-5, 35], color='black', ls='--')
plt.ylabel('Predictions')
plt.xlabel('Real Values')
plt.title('Prediction (Testing Set)')
plt.ylim(-5, 35)
plt.xlim(-5, 35)
plt.grid(True)
plt.show()

# Calculate and print the correlation coefficient, rmse and mae
mae = mean_absolute_error(test_y_reg, test_predictions_scaled)
rmse = np.sqrt(mean_squared_error(test_y_reg, test_predictions_scaled))
correlation_coefficient = np.corrcoef(test_predictions_scaled, test_y_reg.to_numpy())[0, 1]

print("\n--- Resultados da Avaliação no Conjunto de Teste ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("Correlation Coefficient in testing set: %.4f" % correlation_coefficient)



train_predictions = model.predict(X_train_scaled).flatten() # predict radom activities with the built linear regression model

#train_predictions = scaler_y.inverse_transform(train_predictions_scaled).flatten()

plt.scatter(train_y_reg, train_predictions, marker = 'o', c = 'blue')
plt.plot([-5,35], [-5,35], color = 'black', ls = '--')
plt.ylabel('Predictions')
plt.xlabel('Real Values')
plt.title('Linear Regression with Normalized Values (Training Set)')
plt.ylim(-5, 35)
plt.xlim(-5, 35)
plt.axis(True)
plt.show()

# Calculate and print the correlation coefficient
correlation_coefficient = np.corrcoef(train_predictions, train_y_reg.to_numpy())[0, 1]
print("Correlation Coefficient in training set: %.4f" % correlation_coefficient)

# saves the model architecture
try:
    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
    print("Saved to model_architecture.png")
except ImportError:
    print("Error! Graphviz not installed?")