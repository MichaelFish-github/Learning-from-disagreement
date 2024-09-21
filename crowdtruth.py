import pandas as pd
from collections import defaultdict, Counter
import math

SMALL_NUMBER_CONST = 0.00000001

class Metrics:
    @staticmethod
    def unit_quality_score(unit_id, unit_work_ann_dict, wqs, aqs):
        uqs_numerator = 0.0
        uqs_denominator = 0.0
        worker_ids = list(unit_work_ann_dict[unit_id].keys())

        for worker_i in range(len(worker_ids) - 1):
            for worker_j in range(worker_i + 1, len(worker_ids)):
                numerator = 0.0
                denominator_i = 0.0
                denominator_j = 0.0

                worker_i_vector = unit_work_ann_dict[unit_id][worker_ids[worker_i]]
                worker_j_vector = unit_work_ann_dict[unit_id][worker_ids[worker_j]]

                numerator += aqs * (worker_i_vector * worker_j_vector)
                denominator_i += aqs * (worker_i_vector * worker_i_vector)
                denominator_j += aqs * (worker_j_vector * worker_j_vector)

                denominator = math.sqrt(denominator_i * denominator_j)
                if denominator < SMALL_NUMBER_CONST:
                    denominator = SMALL_NUMBER_CONST
                weighted_cosine = numerator / denominator

                uqs_numerator += weighted_cosine * wqs[worker_ids[worker_i]] * wqs[worker_ids[worker_j]]
                uqs_denominator += wqs[worker_ids[worker_i]] * wqs[worker_ids[worker_j]]

        if uqs_denominator < SMALL_NUMBER_CONST:
            uqs_denominator = SMALL_NUMBER_CONST
        return uqs_numerator / uqs_denominator

    @staticmethod
    def worker_quality_score(worker_id, work_unit_ann_dict, unit_ann_dict, uqs, aqs):
        wqs_numerator = 0.0
        wqs_denominator = 0.0
        worker_units = work_unit_ann_dict[worker_id]

        for unit_id in worker_units:
            numerator = 0.0
            denominator_w = 0.0
            denominator_u = 0.0

            worker_annotation = work_unit_ann_dict[worker_id][unit_id]
            unit_annotation = unit_ann_dict[unit_id]

            for ann in unit_annotation:
                numerator += aqs * worker_annotation * (unit_annotation[ann] - worker_annotation)
                denominator_w += aqs * (worker_annotation * worker_annotation)
                denominator_u += aqs * (unit_annotation[ann] * unit_annotation[ann])

            denominator = math.sqrt(denominator_w * denominator_u)
            if denominator < SMALL_NUMBER_CONST:
                denominator = SMALL_NUMBER_CONST
            weighted_cosine = numerator / denominator

            wqs_numerator += weighted_cosine * uqs[unit_id]
            wqs_denominator += uqs[unit_id]

        if wqs_denominator < SMALL_NUMBER_CONST:
            wqs_denominator = SMALL_NUMBER_CONST
        return wqs_numerator / wqs_denominator

    @staticmethod
    def worker_worker_agreement(unit_work_ann_dict):
        wwa = defaultdict(dict)
        worker_ids = list(unit_work_ann_dict.keys())

        for worker_i in range(len(worker_ids) - 1):
            for worker_j in range(worker_i + 1, len(worker_ids)):
                worker_i_id = worker_ids[worker_i]
                worker_j_id = worker_ids[worker_j]

                numerator = 0.0
                denominator_i = 0.0
                denominator_j = 0.0

                for unit_id in unit_work_ann_dict:
                    if worker_i_id in unit_work_ann_dict[unit_id] and worker_j_id in unit_work_ann_dict[unit_id]:
                        worker_i_vector = unit_work_ann_dict[unit_id][worker_i_id]
                        worker_j_vector = unit_work_ann_dict[unit_id][worker_j_id]

                        numerator += worker_i_vector * worker_j_vector
                        denominator_i += worker_i_vector * worker_i_vector
                        denominator_j += worker_j_vector * worker_j_vector

                denominator = math.sqrt(denominator_i * denominator_j)
                if denominator < SMALL_NUMBER_CONST:
                    denominator = SMALL_NUMBER_CONST
                wwa[worker_i_id][worker_j_id] = numerator / denominator

        return wwa

    @staticmethod
    def worker_specific_agreement(worker_id, work_unit_ann_dict, unit_ann_dict):
        wsa_numerator = 0.0
        wsa_denominator = 0.0
        worker_units = work_unit_ann_dict[worker_id]

        for unit_id in worker_units:
            worker_annotation = work_unit_ann_dict[worker_id][unit_id]
            unit_annotation = unit_ann_dict[unit_id]

            for ann in unit_annotation:
                wsa_numerator += worker_annotation * unit_annotation[ann]
                wsa_denominator += worker_annotation * worker_annotation

        if wsa_denominator < SMALL_NUMBER_CONST:
            wsa_denominator = SMALL_NUMBER_CONST
        return wsa_numerator / wsa_denominator
    

def get_single_label_df(df, dataset_name, unit_work_ann_dict, wqs):
        # Calculate the aggregated final annotations per example/unit
    aggregated_annotations = {}
    for unit_id in unit_work_ann_dict:
        total_weight = 0.0
        weighted_sum = 0.0
        for worker_id, annotation in unit_work_ann_dict[unit_id].items():
            weight = wqs[worker_id]
            weighted_sum += weight * annotation
            total_weight += weight
        if total_weight < SMALL_NUMBER_CONST:
            total_weight = SMALL_NUMBER_CONST
        aggregated_annotations[unit_id] = weighted_sum / total_weight

    # print("Aggregated Final Annotations per Unit:", aggregated_annotations)

    # Assign each example in q7_df the aggregated annotation
    crowdtruth_df_single_label = df.copy()
    crowdtruth_df_single_label = crowdtruth_df_single_label.drop_duplicates(subset=['example_id'])
    # remove label, annotator_metadata columns
    crowdtruth_df_single_label = crowdtruth_df_single_label.drop(columns=['label', 'annotator_metadata', 'annotator_id'])
    crowdtruth_df_single_label['label'] = crowdtruth_df_single_label['example_id'].map(aggregated_annotations)
    # Save as csv
    crowdtruth_df_single_label.to_csv(f'{dataset_name}-crowdtruth-scalars.csv', index=False)

def get_vector_label_df(df, dataset_name, unit_work_ann_dict, wqs):
    # Calculate the aggregated annotation distribution per example/unit
    aggregated_annotations = {}
    for unit_id in unit_work_ann_dict:
        # Initialize a dictionary to store weighted votes for each category
        weighted_votes = {0: 0.0, 0.5: 0.0, 1: 0.0}
        total_weight = 0.0

        # Sum the weighted votes for each category
        for worker_id, annotation in unit_work_ann_dict[unit_id].items():
            weight = wqs[worker_id]
            weighted_votes[annotation] += weight
            total_weight += weight

        # Normalize the weighted votes to create a probability distribution
        if total_weight < SMALL_NUMBER_CONST:
            total_weight = SMALL_NUMBER_CONST

        aggregated_annotations[unit_id] = {category: weighted_votes[category] / total_weight for category in weighted_votes}

    # Print the distribution for each example/unit
    # print("Aggregated Annotation Distributions per Unit:", aggregated_annotations)

    # Assign each example in q7_df the aggregated annotation distribution
    crowdtruth_vectors = df.copy()
    crowdtruth_vectors = crowdtruth_vectors.drop_duplicates(subset=['example_id'])
    # remove label, annotator_metadata columns
    crowdtruth_vectors = crowdtruth_vectors.drop(columns=['label', 'annotator_metadata', 'annotator_id'])
    crowdtruth_vectors['label'] = crowdtruth_vectors['example_id'].map(aggregated_annotations)

    # transform label column from dict to list
    crowdtruth_vectors['label'] = crowdtruth_vectors['label'].apply(lambda x: [x[0], x[0.5], x[1]])
    
    # Save as csv
    crowdtruth_vectors.to_csv(f'{dataset_name}-crowdtruth-vectors.csv', index=False)


def main():
    # CSV with columns: example_id, text, annotator_id, label (0 - no, 0.5 - maybe, 1 - yes)
    trinary_df = pd.read_csv('./datasets/measuring-hate-speech/measuring-hate-speech-q7.csv')

    # Enter your dataset's name
    dataset_name = 'measuring-hate-speech'
    
    # if the label column is not in the format mentioned above, you can define the correct mapping in the function below
    map_to_trinary = True
    if map_to_trinary:
        def map_label(label):
            if label in [0, 1]:
                return 0
            elif label == 2:
                return 0.5
            elif label in [3, 4]:
                return 1
        trinary_df = trinary_df.copy()
        trinary_df['label'] = trinary_df['label'].apply(map_label)


    unit_work_ann_dict = defaultdict(dict)
    work_unit_ann_dict = defaultdict(dict)
    unit_ann_dict = defaultdict(Counter)

    for _, row in trinary_df.iterrows():
        unit_id = row['example_id']
        worker_id = row['annotator_id']
        label = row['label']

        unit_work_ann_dict[unit_id][worker_id] = label
        work_unit_ann_dict[worker_id][unit_id] = label
        unit_ann_dict[unit_id][label] += 1

    # Initialize quality scores
    uqs = {unit_id: 1.0 for unit_id in unit_work_ann_dict}
    wqs = {worker_id: 1.0 for worker_id in work_unit_ann_dict}
    aqs = 1.0  # Since the labels are numeric, we can use a single value

    max_delta = 0.001
    iterations = 0

    while max_delta >= 0.001:
        uqs_new = {}
        wqs_new = {}

        max_delta = 0.0

        # Compute new UQS
        for unit_id in unit_work_ann_dict:
            uqs_new[unit_id] = Metrics.unit_quality_score(unit_id, unit_work_ann_dict, wqs, aqs)
            max_delta = max(max_delta, abs(uqs_new[unit_id] - uqs[unit_id]))

        # Compute new WQS
        for worker_id in work_unit_ann_dict:
            wqs_new[worker_id] = Metrics.worker_quality_score(worker_id, work_unit_ann_dict, unit_ann_dict, uqs_new, aqs)
            max_delta = max(max_delta, abs(wqs_new[worker_id] - wqs[worker_id]))

        # Update scores
        uqs = uqs_new
        wqs = wqs_new

        iterations += 1
        print(f"Iteration {iterations} completed; max delta: {max_delta}")

    # Save single label df
    get_single_label_df(trinary_df, dataset_name, unit_work_ann_dict, wqs)
    # Save vector label df
    get_vector_label_df(trinary_df, dataset_name, unit_work_ann_dict, wqs)

    # Save the worker quality scores as csv using pandas
    worker_quality_scores = pd.DataFrame.from_dict(wqs, orient='index', columns=['quality_score'])
    worker_quality_scores.to_csv(f'{dataset_name}-worker-quality-scores.csv')

        # Compute WWA and WSA metrics
    # wwa = Metrics.worker_worker_agreement(unit_work_ann_dict)
    # wsa = {worker_id: Metrics.worker_specific_agreement(worker_id, work_unit_ann_dict, unit_ann_dict) for worker_id in work_unit_ann_dict}

    # Results
    # print("Final Unit Quality Scores:", uqs)
    # print("Final Worker Quality Scores:", wqs)
    # print("Worker-Worker Agreement:", wwa)
    # print("Worker Specific Agreement:", wsa)

if __name__ == '__main__':
    main()