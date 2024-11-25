import kNearestNeighbours as knn
import nearestClassCentroid as ncc
import convolutionalNeuralNetwork as cnn

def main():
    # Calling every algorithm.
    centroid=ncc.NCC("DB")
    knn1=knn.KNN("DB",1)
    knn3=knn.KNN("DB",3)
    cnn.startNetwork(35, 64, 0.0008)

main()