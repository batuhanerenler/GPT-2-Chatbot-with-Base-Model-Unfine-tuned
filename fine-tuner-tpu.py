import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


class GPT2FineTuner(QWidget):
    def __init__(self):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        self.train_button = QPushButton("Train", self)
        self.train_button.clicked.connect(self.fine_tune_gpt2)

        self.progress_label = QLabel("Training Progress:")
        self.progress_bar = QProgressBar(self)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("GPT-2 Fine-tuning"))
        layout.addWidget(self.train_button)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

    

    def fine_tune_gpt2(self):
        def tokenize_function(example):
            question = example["question"]
            context = example["context"]
            answer0 = example["answer0"]
            answer1 = example["answer1"]
            answer2 = example["answer2"]
            answer3 = example["answer3"]
            label = int(example["label"])  # Convert the label value to integer
            correct_answer = [answer0, answer1, answer2, answer3][label]

            question_input = self.tokenizer(question, return_tensors="pt")
            context_input = self.tokenizer(context, return_tensors="pt")
            answer_input = self.tokenizer(correct_answer, return_tensors="pt")

            input_ids = torch.cat([question_input["input_ids"], context_input["input_ids"], answer_input["input_ids"]], dim=1)
            attention_mask = torch.cat([question_input["attention_mask"], context_input["attention_mask"], answer_input["attention_mask"]], dim=1)

            return {"input_ids": input_ids[0], "attention_mask": attention_mask[0]}


        data = load_dataset("cosmos_qa")
        tokenized_texts = data.map(tokenize_function, batched=True, remove_columns=["id", "context", "question", "answer0", "answer1", "answer2", "answer3", "label"])

        config = GPT2Config.from_pretrained("gpt2")
        block_size = config.n_ctx

        training_args = TrainingArguments(
            output_dir="./trained_model",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
            logging_steps=100,
            logging_dir="./logs",
            report_to="none",
            disable_tqdm=True,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_texts["train"],
        )

        self.progress_bar.setValue(0)
        total_steps = len(trainer.train_dataloader())

        def on_log(args, state, control, logs):
            step = state.global_step
            self.progress_bar.setValue(step / total_steps * 100)

        training_args._on_log = on_log
        trainer.train()

        self.model.save_pretrained("./trained_model")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    fine_tuner = GPT2FineTuner()
    fine_tuner.show()
    sys.exit(app.exec_())
