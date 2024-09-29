from tensorflow.keras.metrics import Precision, Recall, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, AUC

class CommonData:
    epochs = 1
    maxlen=70 # Maxium domain name lenght
    batch_size = 50
    verbose = 1
    metrics = ['accuracy', Precision(), Recall(), FalsePositives(), FalseNegatives(), TruePositives(), TrueNegatives(), AUC()]