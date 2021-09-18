import collections
from torch import nn
from transformer import Trainer

class xxxTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Since the model had already
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Outputs should be include the loss/logits/hidden_states
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def inference(self, dev_dataset, batch_size=32):
        total = len(dev_dataset)

        output = liwt
        data_loader = DataLoader(dev_dataset, batch_size=batch_size)
        for batch in data_loader:
           output = self.model(**batch)
           print(output)

