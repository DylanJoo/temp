import argparse

args = ArgumentParser()
parser.add_argument("-lif_json", "--path_lif", type=str)
parser.add_argument("-lif_tsv", "--path_lif_output", type=str)
parser.add_argument("-lag", "--n_previous_question", default=1, type=int)
parser.add_argument("-answer", "--include_answer", action='store_true', default=False)
args = parser.parse_args()


def convert_lif_to_clf_task(args):

    lif = json.load(open(args.path_lif, 'r'))
    lif_data = lif['data']
    fout = open(args.path_lif_output, 'w')

    for i_topic, topic enumerate(lif_data):

        content = topic['paragraphs'][0]

        for i_turn, turn in enumerate(content['qas']):

            # Question i 
            question = turn['candidate']

            # Label of follow-up
            if turn['label'] == 0:
                followup = "true"
            else:
                followup = "false"

            # Question before i
            question_prev = turn['prev_qs'][-(args.n_previous_question):]

            # Answer(response) before i
            if args.include_answer:
                answer_prev = turn['prev_ans'][-(args.n_previous_question):]

            # Context(Q and A) before i
            qa_prev = list()
            for i, q in enumerate(question_prev):
                qa.append(q)
                if args.answer:
                    qa.append(answer_prev[i])

            qa_prev = " ||| ".join(qa_prev)

            fout.write("Question: {} History: {} Follow-up:\t{}\n".format(
                question, qa_prev, followup) 


convert_lif_to_clf_task(args)
print("DONE")

