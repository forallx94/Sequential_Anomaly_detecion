from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import pandas as pd

def one_class_svm(X_train):
    clf = OneClassSVM(gamma='auto').fit(X_train)
    pred = clf.predict(X_train)
    pred_score = clf.score_samples(X_train)

    pca = PCA(n_components=2)
    pca.fit(X_train)
    printcipalComponents = pca.transform(X_train)
    principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = [1, -1]
    colors = ['g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = pred == target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component1']
                , principalDf.loc[indicesToKeep, 'principal component2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()

    plt.savefig('./result/One_class_SVM_PCA.png')

    return clf