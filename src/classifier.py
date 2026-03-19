"""Resume–JD fit classifier supporting logistic regression, random forest, and BERT.

All heavy ML imports are wrapped in try/except blocks so the module can be
imported even in lightweight environments.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    BERT_MODEL,
    CLASSIFICATION_LABELS,
    CLASSIFICATION_THRESHOLDS,
    MODEL_DIR,
)
from src.data_collection import JobDescription, Resume

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.metrics import accuracy_score, f1_score, classification_report  # type: ignore
    from sklearn.preprocessing import LabelEncoder  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — LR/RF classifiers unavailable.")

try:
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("torch/transformers not installed — BERT classifier unavailable.")


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------


def _build_feature_text(resume: Resume, jd: JobDescription) -> str:
    """Concatenate resume + JD into a single string for text-based models."""
    resume_part = resume.clean_text or resume.raw_text or ""
    jd_part = jd.raw_text or ""
    skills_part = " ".join(resume.skills)
    required_part = " ".join(jd.required_skills)
    return f"{resume_part} [SEP] {jd_part} [SKILLS] {skills_part} [REQUIRED] {required_part}"


def _score_to_label_index(score: float) -> int:
    """Convert a composite score to a class index.

    Classes:
    - 0: Strong Match
    - 1: Moderate Match
    - 2: Weak Match
    - 3: No Match
    """
    if score >= CLASSIFICATION_THRESHOLDS["strong"]:
        return 0
    if score >= CLASSIFICATION_THRESHOLDS["moderate"]:
        return 1
    if score >= CLASSIFICATION_THRESHOLDS["weak"]:
        return 2
    return 3


# ---------------------------------------------------------------------------
# BERT Dataset (torch)
# ---------------------------------------------------------------------------


def _make_bert_dataset(
    texts_a: List[str],
    texts_b: List[str],
    labels: List[int],
    tokenizer: Any,
    max_len: int = 512,
) -> Any:
    """Create a torch Dataset for BERT fine-tuning."""
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("torch/transformers required for BERT dataset.")
    from torch.utils.data import Dataset  # type: ignore

    encodings = tokenizer(
        texts_a,
        texts_b,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    tensor_labels = torch.tensor(labels, dtype=torch.long)

    class _ResumeDataset(Dataset):
        def __len__(self) -> int:
            return len(tensor_labels)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            item = {k: v[idx] for k, v in encodings.items()}
            item["labels"] = tensor_labels[idx]
            return item

    return _ResumeDataset()


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


class ResumeClassifier:
    """Resume–JD fit classifier with multiple backend options.

    Supported model types:
    - ``"logistic_regression"``: TF-IDF + Logistic Regression (fast, low memory)
    - ``"random_forest"``: TF-IDF + Random Forest (interpretable)
    - ``"bert"``: Fine-tuned BERT sequence classifier (highest accuracy)
    """

    def __init__(self, model_type: str = "logistic_regression") -> None:
        """Initialise the classifier.

        Args:
            model_type: One of ``"logistic_regression"``, ``"random_forest"``,
                or ``"bert"``.
        """
        if model_type not in ("logistic_regression", "random_forest", "bert"):
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                "Choose: logistic_regression, random_forest, bert."
            )

        self.model_type = model_type
        self._sklearn_model: Optional[Any] = None
        self._bert_model: Optional[Any] = None
        self._bert_tokenizer: Optional[Any] = None
        self._is_trained: bool = False

        if model_type in ("logistic_regression", "random_forest") and not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for LR/RF classifiers.")
        if model_type == "bert" and not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("torch/transformers are required for BERT classifier.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: List[Tuple[Resume, JobDescription, int]],
        val_data: Optional[List[Tuple[Resume, JobDescription, int]]] = None,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Train the classifier on labelled (resume, JD, label) triplets.

        Args:
            train_data: List of ``(resume, jd, label_index)`` tuples.
                Label indices: 0=Strong, 1=Moderate, 2=Weak, 3=No Match.
            val_data: Optional validation set in the same format.
            output_dir: Directory to save the trained model artefacts.
                Defaults to ``<MODEL_DIR>/<model_type>/``.
            **kwargs: Extra keyword arguments forwarded to the underlying
                trainer (e.g. ``num_epochs``, ``learning_rate`` for BERT).

        Returns:
            Dict with training metrics (``accuracy``, ``f1_macro``, etc.).
        """
        save_dir = Path(output_dir) if output_dir else MODEL_DIR / self.model_type
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model_type == "bert":
            return self._train_bert(train_data, val_data, str(save_dir), **kwargs)
        return self._train_sklearn(train_data, val_data, str(save_dir))

    def _train_sklearn(
        self,
        train_data: List[Tuple[Resume, JobDescription, int]],
        val_data: Optional[List[Tuple[Resume, JobDescription, int]]],
        save_dir: str,
    ) -> Dict[str, float]:
        """Train a TF-IDF + LR/RF pipeline."""
        texts = [_build_feature_text(r, jd) for r, jd, _ in train_data]
        labels = [label for _, _, label in train_data]

        clf = (
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            if self.model_type == "logistic_regression"
            else RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)
        )

        self._sklearn_model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)),
            ("clf", clf),
        ])
        self._sklearn_model.fit(texts, labels)
        self._is_trained = True

        # Save model
        model_path = Path(save_dir) / "model.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(self._sklearn_model, fh)
        logger.info("Saved %s model to %s", self.model_type, model_path)

        # Training metrics
        train_preds = self._sklearn_model.predict(texts)
        metrics: Dict[str, float] = {
            "train_accuracy": float(accuracy_score(labels, train_preds)),
            "train_f1_macro": float(f1_score(labels, train_preds, average="macro")),
        }

        if val_data:
            val_texts = [_build_feature_text(r, jd) for r, jd, _ in val_data]
            val_labels = [label for _, _, label in val_data]
            val_preds = self._sklearn_model.predict(val_texts)
            metrics["val_accuracy"] = float(accuracy_score(val_labels, val_preds))
            metrics["val_f1_macro"] = float(f1_score(val_labels, val_preds, average="macro"))

        logger.info("Training complete: %s", metrics)
        return metrics

    def _train_bert(
        self,
        train_data: List[Tuple[Resume, JobDescription, int]],
        val_data: Optional[List[Tuple[Resume, JobDescription, int]]],
        save_dir: str,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> Dict[str, float]:
        """Fine-tune a BERT model on resume–JD pairs."""
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL, num_labels=len(CLASSIFICATION_LABELS)
        )

        resume_texts = [r.clean_text[:512] for r, _, _ in train_data]
        jd_texts = [jd.raw_text[:256] for _, jd, _ in train_data]
        labels = [lbl for _, _, lbl in train_data]
        train_dataset = _make_bert_dataset(resume_texts, jd_texts, labels, tokenizer)

        eval_dataset = None
        if val_data:
            v_resumes = [r.clean_text[:512] for r, _, _ in val_data]
            v_jds = [jd.raw_text[:256] for _, jd, _ in val_data]
            v_labels = [lbl for _, _, lbl in val_data]
            eval_dataset = _make_bert_dataset(v_resumes, v_jds, v_labels, tokenizer)

        training_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="eval_accuracy",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        def compute_metrics(eval_pred: Any) -> Dict[str, float]:
            logits, lbl = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(lbl, preds)),
                "f1_macro": float(f1_score(lbl, preds, average="macro")),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if _NUMPY_AVAILABLE else None,
        )

        result = trainer.train()
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

        self._bert_model = model
        self._bert_tokenizer = tokenizer
        self._is_trained = True

        logger.info("BERT training complete: %s", result.metrics)
        return result.metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        resume: Resume,
        jd: JobDescription,
    ) -> Dict[str, Any]:
        """Predict the match class for a resume–JD pair.

        Args:
            resume: Parsed resume.
            jd: Parsed job description.

        Returns:
            Dict with keys:
            - ``class_label`` (str): Predicted match class.
            - ``class_index`` (int): Class index [0–3].
            - ``probabilities`` (dict[str, float]): Per-class probabilities.
            - ``confidence`` (float): Probability of predicted class.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier has not been trained yet. Call train() first.")

        if self.model_type == "bert":
            return self._predict_bert(resume, jd)
        return self._predict_sklearn(resume, jd)

    def _predict_sklearn(
        self, resume: Resume, jd: JobDescription
    ) -> Dict[str, Any]:
        text = _build_feature_text(resume, jd)
        proba = self._sklearn_model.predict_proba([text])[0]
        class_index = int(np.argmax(proba))
        return {
            "class_label": CLASSIFICATION_LABELS[class_index],
            "class_index": class_index,
            "probabilities": {
                CLASSIFICATION_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            },
            "confidence": round(float(proba[class_index]), 4),
        }

    def _predict_bert(
        self, resume: Resume, jd: JobDescription
    ) -> Dict[str, Any]:
        if self._bert_model is None or self._bert_tokenizer is None:
            raise RuntimeError("BERT model not loaded. Train or load first.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._bert_model.to(device)
        self._bert_model.eval()

        inputs = self._bert_tokenizer(
            resume.clean_text[:512],
            jd.raw_text[:256],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._bert_model(**inputs).logits
        proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        class_index = int(np.argmax(proba))

        return {
            "class_label": CLASSIFICATION_LABELS[class_index],
            "class_index": class_index,
            "probabilities": {
                CLASSIFICATION_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            },
            "confidence": round(float(proba[class_index]), 4),
        }

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        resume: Resume,
        jd: JobDescription,
        method: str = "lime",
        top_features: int = 10,
    ) -> Dict[str, Any]:
        """Generate an explanation for a prediction using LIME or feature weights.

        For BERT, falls back to a heuristic feature explanation when LIME is
        not available.

        Args:
            resume: Parsed resume.
            jd: Parsed job description.
            method: Explanation method (``"lime"`` or ``"shap"``).
            top_features: Number of top features to include.

        Returns:
            Dict with ``prediction``, ``probabilities``, ``top_features``,
            and ``explanation_text``.
        """
        prediction_result = self.predict(resume, jd)
        features: List[Tuple[str, float]] = []

        if self.model_type in ("logistic_regression", "random_forest") and method == "lime":
            try:
                from lime.lime_text import LimeTextExplainer  # type: ignore

                explainer = LimeTextExplainer(class_names=CLASSIFICATION_LABELS)
                text = _build_feature_text(resume, jd)

                def predict_fn(texts: List[str]) -> Any:
                    return self._sklearn_model.predict_proba(texts)

                explanation = explainer.explain_instance(
                    text, predict_fn, num_features=top_features, num_samples=300
                )
                features = explanation.as_list()
            except ImportError:
                logger.warning("LIME not installed — skipping feature explanation.")
            except Exception as exc:
                logger.warning("LIME explanation failed: %s", exc)

        elif self.model_type == "logistic_regression" and not features:
            # Coefficient-based explanation for LR
            try:
                vectorizer = self._sklearn_model.named_steps["tfidf"]
                clf = self._sklearn_model.named_steps["clf"]
                text = _build_feature_text(resume, jd)
                vec = vectorizer.transform([text])
                feature_names = vectorizer.get_feature_names_out()
                class_idx = prediction_result["class_index"]
                coefs = clf.coef_[class_idx]
                top_idx = np.argsort(np.abs(coefs * vec.toarray()[0]))[::-1][:top_features]
                features = [
                    (feature_names[i], float(coefs[i] * vec.toarray()[0][i]))
                    for i in top_idx
                ]
            except Exception as exc:
                logger.warning("Coefficient explanation failed: %s", exc)

        explanation_text = self._generate_explanation_text(
            prediction_result["class_label"], features, resume, jd
        )

        return {
            "prediction": prediction_result,
            "top_features": features,
            "explanation_text": explanation_text,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Optional[str] = None) -> str:
        """Save the trained model to disk.

        Args:
            output_dir: Directory path. Defaults to ``<MODEL_DIR>/<model_type>/``.

        Returns:
            Absolute path of the saved directory.
        """
        save_path = Path(output_dir) if output_dir else MODEL_DIR / self.model_type
        save_path.mkdir(parents=True, exist_ok=True)

        if self.model_type in ("logistic_regression", "random_forest"):
            with (save_path / "model.pkl").open("wb") as fh:
                pickle.dump(self._sklearn_model, fh)
        elif self.model_type == "bert" and self._bert_model is not None:
            self._bert_model.save_pretrained(str(save_path))
            self._bert_tokenizer.save_pretrained(str(save_path))

        logger.info("Model saved to %s", save_path)
        return str(save_path)

    def load(self, model_dir: str) -> None:
        """Load a previously saved model from disk.

        Args:
            model_dir: Directory containing model artefacts.
        """
        path = Path(model_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Model directory not found: {model_dir}")

        if self.model_type in ("logistic_regression", "random_forest"):
            model_file = path / "model.pkl"
            if not model_file.exists():
                raise FileNotFoundError(f"model.pkl not found in {model_dir}")
            with model_file.open("rb") as fh:
                self._sklearn_model = pickle.load(fh)
        elif self.model_type == "bert":
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("torch/transformers required to load BERT model.")
            self._bert_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._bert_model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self._is_trained = True
        logger.info("Model loaded from %s", model_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_explanation_text(
        label: str,
        features: List[Tuple[str, float]],
        resume: Resume,
        jd: JobDescription,
    ) -> str:
        """Produce a human-readable explanation string."""
        lines = [f"Prediction: **{label}**"]

        if features:
            positive = [f[0] for f in features if f[1] > 0][:3]
            negative = [f[0] for f in features if f[1] < 0][:3]
            if positive:
                lines.append(f"Key matching signals: {', '.join(positive)}")
            if negative:
                lines.append(f"Potential gaps: {', '.join(negative)}")
        else:
            # Heuristic explanation
            matched = set(s.lower() for s in resume.skills) & set(
                s.lower() for s in jd.required_skills
            )
            missing = set(s.lower() for s in jd.required_skills) - set(
                s.lower() for s in resume.skills
            )
            if matched:
                lines.append(f"Matched skills: {', '.join(list(matched)[:5])}")
            if missing:
                lines.append(f"Missing skills: {', '.join(list(missing)[:5])}")

        return ". ".join(lines)
