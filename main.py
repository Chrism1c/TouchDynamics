# Main script

"""
# DATASETS #

DSN-2009 (.tie5Roanl): https://www.cs.cmu.edu/~keystroke/
(H, DD, UP)*(10 caratteri + ENTER)
Keystroke100 (try4-mbs): https://personal.ie.cuhk.edu.hk/~ccloy/downloads_keystroke100.html
Buffalo (Steave Jobs text):  https://www.buffalo.edu/cubs/research/datasets.html

"""

"""
# Systems #

# Manhattan : Simple, Filtered, Scaled
# MahalanobisDetector
# one-class SVM
# GMM : spherical, diag (Correggere EER)
# K-means

# NN su : D, H, DD, total, pca3, pca10

# LSTM (RNN)


TODO:
1 Strutturare i dataset Buffalo e Keystroke100 in 'dataframe.pickle'
2 Codificare un estrattore di Features nel formato DSN-2009 per Keystroke100 e Paper1 per Buffalo
3 Addestrare e Testare le Reti sugli altri Dataset

4* Keylogger per generare un dataset di keystrokes
5* Online Auth (Sliding Window)

# 6* WiFi Quack (invia keystrokes a delay e ritmo standard dal cellulare ad arduino collegato al PC)
# 7* Replicare il Keystroke dynamics di un Utente e scrivere fingendosi lui

"""

if __name__ == '__main__':
    print('PyCharm')
