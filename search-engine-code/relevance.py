import math
import csv
import math

def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    if not search_result_relevances:
        return 0.0

    cumulative_precisions = []
    relevant_count = 0

    total_relevant = sum(1 for score in search_result_relevances if score == 1)

    if total_relevant == 0:
        return 0.0

    for idx, is_relevant in enumerate(search_result_relevances[:cut_off], 1):
        if is_relevant:
            relevant_count += 1
            precision_at_rank = relevant_count / idx
            cumulative_precisions.append(precision_at_rank)

    average_precision = sum(cumulative_precisions) / total_relevant
    return average_precision


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10) -> float:
    def dcg(relevances, cut_off):
        dcg_val = relevances[0]
        for idx, rel in enumerate(relevances[1:cut_off], 2):
            dcg_val += rel / math.log2(idx)
        return dcg_val
    
    actual_dcg = dcg(search_result_relevances, cut_off)
    ideal_dcg = dcg(ideal_relevance_score_ordering, cut_off)
    
    if ideal_dcg == 0:
        return 0.0
    
    ndcg_val = actual_dcg / ideal_dcg
    return ndcg_val


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded

        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset

    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    return {'map': 0, 'ndcg': 0}



