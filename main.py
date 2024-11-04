import kNearestNeighbours as knn
import nearestClassCentroid as ncc

def compare(NCC_acc, KNN_acc):
    if (NCC_acc-KNN_acc>0):
        print(f"Nearest class centroid method was more accurate by: {(NCC_acc-KNN_acc) * 100:.2f}%")
    elif (NCC_acc-KNN_acc<0):
        print(f"K nearest neighbour method was more accurate by: {(KNN_acc-NCC_acc) * 100:.2f}%")
    else:
        print(f"Both methods had equal accuracies with: {NCC_acc * 100:.2f}%")

compare(ncc.NCC("DB"),knn.KNN("DB",1))