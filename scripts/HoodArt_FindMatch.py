# Chujun May 8, 2023

# Import libraries
import numpy
import numba
import random

# Define a function for Fast cosine distance computation using numba 
# compute the cosine distance between a vector u and all rows in a matrix M
@numba.guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", target='parallel')
def fast_cosine_gufunc(u, v, result):
    m = u.shape[0] # how many features
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m): # for each element (feature) in the vector
        if (numpy.isnan(u[i])) or (numpy.isnan(v[i])):
            continue

        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = numpy.sqrt(u_norm)
    v_norm = numpy.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    result[:] = ratio

# Write a function that takes in a target text embedding and storage of embeddings of old text
# compute the cosine SIMILARITY between this target text embedding with every row in the old text embedding storage
# n_feature = 768 for ALBERT
def RowWiseDistance(target_embedding, storage_embedding, n_feature):
    # need to match the u shape in the fast_cosine_gufunc
    target_embedding = numpy.reshape(target_embedding, (n_feature,)) 
    # need to match the float64 data type defined in the fast_cosine_gufunc
    target_embedding = numpy.float64(target_embedding) 
    # need to match the float64 data type defined in the fast_cosine_gufunc
    storage_embedding = numpy.float64(storage_embedding) 
    # compute the cosine distance between the target embedding and every row of the storage embedding
    similarity = fast_cosine_gufunc(target_embedding, storage_embedding)
    return similarity

# Write a function to pick out a randomly selected row in the storage that is among the top K most similar embedding to the target embedding
# This function takes in the similarity output from the function RowWiseDistance(target_embedding, storage_embedding)
# Returns the row ID within the storage_embedding of the selected row
def PickTopSimilar(similarity_vector, top_k):
    # sort the similarity array
    sorted_index_array = numpy.argsort(similarity_vector)
    sorted_array = similarity_vector[sorted_index_array]
    # get the largest top k values
    rslt = sorted_array[-top_k : ]
    # randomly pick one from the largest k values
    selected = random.choice(list(rslt))
    # find the index of this selected value in the orginal simlarity vector
    selected_index = list(similarity_vector).index(selected)
    return selected_index








