import kNearestNeighbours as knn
import nearestClassCentroid as ncc

# Simple function to compare the results of the algorithms and display them accordingly as well as the exact percentage difference.
def compare(NCC_acc, KNN_acc):
    if (NCC_acc-KNN_acc>0):
        print(f"Nearest class centroid method was more accurate by: {(NCC_acc-KNN_acc) * 100:.2f}%")
    elif (NCC_acc-KNN_acc<0):
        print(f"K nearest neighbour method was more accurate by: {(KNN_acc-NCC_acc) * 100:.2f}%")
    else:
        print(f"Both methods had equal accuracies with: {NCC_acc * 100:.2f}%")

def main():
    # This takes the results of the algorithms in order to compare them
    centroid=ncc.NCC("DB")
    knn1=knn.KNN("DB",1)
    knn3=knn.KNN("DB",3)
    print("\nFor 1st nearest neighbour:")
    compare(centroid,knn1)
    print("For the 3 nearest neighbours:")
    compare(centroid,knn3)

main()