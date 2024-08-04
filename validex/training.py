"""Experimental training mixin for local model training."""

import json
from typing import Any

from validex.logger import log


class TrainingMixin:
    def save(self, filename: str) -> None:
        import torch

        log.info(f"Saving model state to {filename}")
        # Ensure the model exists
        if not hasattr(self, "model"):
            raise AttributeError("Model not found. Please run fit() before saving.")
        # Save the model
        torch.save(self.model.state_dict(), filename)
        log.info(f"Model state saved to {filename}")

    def infer_extract(self, text: str) -> Any:
        import json

        import torch

        log.info("Performing inference extraction")
        if not hasattr(self, "model") or not hasattr(self, "tokenizer"):
            raise AttributeError(
                "Model or tokenizer not found. Please run fit() before inference."
            )

        # Check if MPS is available, otherwise use CPU
        if torch.backends.mps.is_available():
            try:
                device = torch.device("mps")
                self.model.to(device)
            except Exception as e:
                log.warning(f"Error when trying to use MPS: {e}. Falling back to CPU.")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        log.info(f"Using device: {device}")

        inputs = self.tokenizer(text, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=200)

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                parsed_result = json.loads(result)
                log.info("Inference extraction completed")
                return parsed_result
            except json.JSONDecodeError:
                log.warning("Could not parse inference result as JSON")
                return result
        except RuntimeError as e:
            log.error(f"RuntimeError during inference: {e}")
            log.info("Attempting inference on CPU")

            # Fallback to CPU
            self.model.to(torch.device("cpu"))
            inputs = inputs.to(torch.device("cpu"))

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=200)

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                parsed_result = json.loads(result)
                log.info("Inference extraction completed on CPU")
                return parsed_result
            except json.JSONDecodeError:
                log.warning("Could not parse inference result as JSON")
                return result

    def fit(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_length: int = 512,
        model_name: str = "gpt2",
        output_dir: str = "./results",
        logging_dir: str = "./logs",
        save_total_limit: int = 2,
    ) -> None:
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        log.info("Starting local model training")

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare dataset
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples["src"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            targets = self.tokenizer(
                examples["target"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            inputs["labels"] = targets["input_ids"]
            return inputs

        train_data = [
            {"src": src, "target": json.dumps([s.dict() for s in structs])}
            for src, structs in self.dataset
        ]

        dataset = Dataset.from_list(train_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=logging_dir,
            learning_rate=learning_rate,
            save_total_limit=save_total_limit,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()

        log.info("Local model training completed")
