import math
import time
import csv
import io
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np



def detect_csv_delimiter(file_content, sample_size=5):
    try:
        lines = file_content.split('\n')[:sample_size]
        sample = '\n'.join(lines)

        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        return delimiter
    except:
        common_delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}

        for delim in common_delimiters:
            first_line = file_content.split('\n')[0]
            delimiter_counts[delim] = first_line.count(delim)

        if max(delimiter_counts.values()) > 0:
            return max(delimiter_counts, key=delimiter_counts.get)

        return ','



class ACRULE:
    """Represents a single association rule: X -> Y"""

    def __init__(self, X, Y):
        self.antecedent = X
        self.consequent = Y
        self.antecedentSUP = 0.0
        self.antecedentNConsequentSUP = 0.0
        self.confidence = 0.0

    def toString(self):
        return f"{self.antecedent} ===> {self.consequent} (supportX = {self.antecedentSUP}, supportXY = {self.antecedentNConsequentSUP}, confidence = {self.confidence})"


def notPruneIf_has_all_k_subsets_in_Lk(c: List[str], Lk: Dict[tuple, int]) -> bool:
    """Check Apriori property: all k-subsets are frequent"""
    for idx in range(len(c)):
        subset = tuple(sorted(c[:idx] + c[idx+1:]))
        if subset not in Lk:
            return False
    return True


class ARClassification:
    """Association Rule Generation using Apriori algorithm"""

    def __init__(self, transactions, className, classValues, minsup=0.1, minconf=0.8):
        self.T = transactions
        self.minsup = minsup
        self.min_cnt = int(math.ceil(minsup * len(transactions)))
        self.minconf = minconf
        self.className = className
        self.classValues = classValues
        self.allRules = []

    def ConsideringRules(self, allCandidateRules):
        """Count support and confidence, separate Fk and qualified rules"""
        for rule in allCandidateRules:
            rule.antecedentSUP = 0
            rule.antecedentNConsequentSUP = 0
            for t in self.T:
                if all(item in t for item in rule.antecedent):
                    rule.antecedentSUP += 1
                if all(item in t for item in rule.antecedent + rule.consequent):
                    rule.antecedentNConsequentSUP += 1

            rule.confidence = rule.antecedentNConsequentSUP / (rule.antecedentSUP + 1e-20)

        Fk = []
        for rule in allCandidateRules:
            if rule.antecedentSUP >= self.min_cnt and rule.confidence >= self.minconf:
                self.allRules.append(rule)
            elif rule.antecedentSUP >= self.min_cnt:
                rule.antecedent.sort()
                Fk.append(rule)

        return Fk

    def FindF1Rules(self):
        """Find 1-item association rules"""
        allF1Itemsets = set()
        allCandidateRules = []

        for t in self.T:
            for it in t:
                if not any(it == f'{self.className}:{v}' for v in self.classValues):
                    allF1Itemsets.add(it)

        for item in allF1Itemsets:
            for value in self.classValues:
                rule = ACRULE(X=[item], Y=[f'{self.className}:{value}'])
                allCandidateRules.append(rule)

        return self.ConsideringRules(allCandidateRules)

    def FindFkRules(self, FKRules, k):
        """Generate k-item rules from (k-1)-item rules"""
        Lk: Dict[tuple, int] = {}
        for rule in FKRules:
            Lk[tuple(sorted(rule.antecedent))] = rule.antecedentSUP

        ruleKPlus1 = []
        candidateSet = set()

        for rule1 in FKRules:
            for rule2 in FKRules:
                if rule1.consequent == rule2.consequent:
                    newItemset = list(set(rule1.antecedent + rule2.antecedent))
                    if len(newItemset) != k + 1:
                        continue
                    key = tuple(sorted(newItemset))
                    if key in candidateSet:
                        continue
                    candidateSet.add(key)
                    if notPruneIf_has_all_k_subsets_in_Lk(newItemset, Lk):
                        newRule = ACRULE(X=newItemset, Y=rule2.consequent)
                        ruleKPlus1.append(newRule)

        ruleKPlus1 = self.ConsideringRules(ruleKPlus1)
        return [r for r in ruleKPlus1 if len(r.antecedent) == k + 1]

    def RuleGeneration(self):
        """Generate all association rules"""
        Lk = self.FindF1Rules()
        k = 1
        while Lk:
            Lk = self.FindFkRules(Lk, k)
            k += 1
        for r in self.allRules:
            print(r.toString())


class ACRClassifier:
    """
    Association Rule Classifier with Confidence-Based Weighting

    This classifier generates association rules from training data and uses
    confidence-based voting for predictions.
    """

    def __init__(self, minsup=0.01, minconf=0.5):
        self.minsup = minsup
        self.minconf = minconf
        self.rules = []
        self.className = None
        self.classValues = None
        self.majority_class = None

    def _dataframe_to_transactions(self, X, y=None):
        """Convert binary DataFrame to transaction format"""
        transactions = []
        for idx in range(len(X)):
            transaction = []
            for col in X.columns:
                if X.iloc[idx][col] == 1:
                    transaction.append(f"{col}:1")
            if y is not None:
                transaction.append(f"{self.className}:{int(y.iloc[idx])}")
            transactions.append(transaction)
        return transactions

    def fit(self, X_train, y_train, className='target', progress_callback=None):
        def report_progress(message):
            if progress_callback:
                progress_callback(message)
            print(message)

        self.className = className

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]

        self.classValues = sorted(y_train.unique())
        self.majority_class = y_train.mode()[0]

        report_progress(f"Converting {len(X_train)} samples to transaction format...")
        transactions = self._dataframe_to_transactions(X_train, y_train)
        report_progress(f"Created {len(transactions)} transactions")

        report_progress("Initializing association rule mining...")
        ar_model = ARClassification(
            transactions=transactions,
            className=self.className,
            classValues=self.classValues,
            minsup=self.minsup,
            minconf=self.minconf
        )

        report_progress("Generating 1-itemset rules...")
        Lk = ar_model.FindF1Rules()
        report_progress(f"Found {len(Lk)} frequent 1-itemsets, {len(ar_model.allRules)} rules")

        k = 1
        while Lk:
            k += 1
            report_progress(f"Generating {k}-itemset rules...")
            Lk = ar_model.FindFkRules(Lk, k)
            if Lk:
                report_progress(f"Found {len(Lk)} frequent {k}-itemsets, {len(ar_model.allRules)} total rules")

        self.rules = ar_model.allRules
        report_progress(f"Training complete: Generated {len(self.rules)} association rules")

        return self

    def predict(self, X_test):
        predictions = []

        for idx in range(len(X_test)):
            instance = []
            for col in X_test.columns:
                if X_test.iloc[idx][col] == 1:
                    instance.append(f"{col}:1")

            class_scores = {cls: 0.0 for cls in self.classValues}

            for rule in self.rules:
                if all(item in instance for item in rule.antecedent):
                    class_label = int(rule.consequent[0].split(':')[1])
                    class_scores[class_label] += rule.confidence

            if max(class_scores.values()) > 0:
                predicted_class = max(class_scores, key=class_scores.get)
            else:
                predicted_class = self.majority_class

            predictions.append(predicted_class)

        return np.array(predictions)

    def predict_single(self, instance_dict):
        instance = []
        for feature, value in instance_dict.items():
            if value == 1:
                instance.append(f"{feature}:1")

        class_scores = {cls: 0.0 for cls in self.classValues}
        matching_rules = []

        for rule in self.rules:
            if all(item in instance for item in rule.antecedent):
                class_label = int(rule.consequent[0].split(':')[1])
                class_scores[class_label] += rule.confidence
                matching_rules.append(rule)

        if max(class_scores.values()) > 0:
            predicted_class = max(class_scores, key=class_scores.get)
        else:
            predicted_class = self.majority_class

        return {
            'prediction': predicted_class,
            'class_scores': class_scores,
            'matching_rules': matching_rules
        }

    def get_rule_summary(self):
        """Print summary of generated rules"""
        if not self.rules:
            print("No rules generated yet. Call fit() first.")
            return

        print(f"Association Rule Summary: {len(self.rules)} total rules")

        for class_val in self.classValues:
            class_label = f"{self.className}:{class_val}"
            class_rules = [r for r in self.rules if r.consequent[0] == class_label]
            print(f"\nClass {class_val}: {len(class_rules)} rules")

            sorted_rules = sorted(class_rules, key=lambda r: r.confidence, reverse=True)
            for i, rule in enumerate(sorted_rules[:5]):
                print(f"  {i+1}. {rule.toString()}")

    def get_all_features_in_rules(self):
        features = set()
        for rule in self.rules:
            for item in rule.antecedent:
                feature = item.split(':')[0]
                features.add(feature)
        return features



class DataPreprocessor:
    """
    Preprocesses datasets for Association Rule Classification

    Converts continuous and categorical features to binary format while
    maintaining metadata for UI controls and interpretation.
    """

    def __init__(self):
        self.feature_metadata = {}
        self.target_column = None
        self.original_columns = []
        self.binary_columns = []

    def preprocess_to_binary(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
        def report_progress(message):
            if progress_callback:
                progress_callback(message)
            print(message)

        self.target_column = target_col
        self.original_columns = [col for col in df.columns if col != target_col]
        df_binary = df.copy()

        report_progress("Detecting feature types...")
        continuous_cols = []
        categorical_cols = []
        binary_cols = []

        for col in self.original_columns:
            unique_vals = df[col].nunique()
            if unique_vals == 2 and set(df[col].dropna().unique()).issubset({0, 1}):
                binary_cols.append(col)
            elif df[col].dtype in ['int64', 'float64'] and unique_vals > 10:
                continuous_cols.append(col)
            else:
                categorical_cols.append(col)

        report_progress(f"Detected: {len(continuous_cols)} continuous, {len(categorical_cols)} categorical, {len(binary_cols)} binary features")

        if continuous_cols:
            report_progress("Cleaning outliers...")
        original_len = len(df_binary)
        for col in continuous_cols:
            Q1 = df_binary[col].quantile(0.01)
            Q3 = df_binary[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_binary = df_binary[(df_binary[col] >= lower_bound) & (df_binary[col] <= upper_bound)]

        removed = original_len - len(df_binary)
        if removed > 0:
            report_progress(f"Removed {removed} outliers ({removed/original_len*100:.2f}%)")

        report_progress("Handling missing values...")
        missing_count = 0
        for col in continuous_cols:
            if df_binary[col].isnull().any():
                median_val = df_binary[col].median()
                df_binary[col] = df_binary[col].fillna(median_val)
                missing_count += 1

        for col in categorical_cols + binary_cols:
            if df_binary[col].isnull().any():
                mode_val = df_binary[col].mode()[0] if not df_binary[col].mode().empty else 0
                df_binary[col] = df_binary[col].fillna(mode_val)
                missing_count += 1

        if missing_count > 0:
            report_progress(f"Filled missing values in {missing_count} features")
        else:
            report_progress("No missing values found")

        self.feature_metadata = {}

        if continuous_cols:
            report_progress(f"Converting {len(continuous_cols)} continuous features to binary...")
        for col in continuous_cols:
            mean_val = df_binary[col].mean()
            min_val = df_binary[col].min()
            max_val = df_binary[col].max()
            std_val = df_binary[col].std()

            self.feature_metadata[col] = {
                'type': 'continuous',
                'mean': mean_val,
                'min': min_val,
                'max': max_val,
                'std': std_val,
                'threshold': mean_val,
                'original_name': col
            }

            df_binary[col] = (df_binary[col] > mean_val).astype(int)

        if continuous_cols:
            report_progress(f"Converted {len(continuous_cols)} continuous features")

        if categorical_cols:
            report_progress(f"One-hot encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            categories = sorted(df_binary[col].unique())
            self.feature_metadata[col] = {
                'type': 'categorical',
                'categories': categories,
                'original_name': col
            }

            dummies = pd.get_dummies(df_binary[col], prefix=col, drop_first=False).astype(int)

            for dummy_col in dummies.columns:
                category_value = dummy_col.replace(f"{col}_", "")
                self.feature_metadata[dummy_col] = {
                    'type': 'categorical_dummy',
                    'parent': col,
                    'category': category_value,
                    'original_name': col
                }

            df_binary = pd.concat([df_binary, dummies], axis=1)
            df_binary.drop(col, axis=1, inplace=True)

        if categorical_cols:
            report_progress(f"Created dummy variables for {len(categorical_cols)} categorical features")

        if binary_cols:
            report_progress(f"Processing {len(binary_cols)} binary features...")
        for col in binary_cols:
            self.feature_metadata[col] = {
                'type': 'binary',
                'original_name': col
            }

        self.binary_columns = [col for col in df_binary.columns if col != target_col]

        report_progress(f"Preprocessing complete: {len(self.binary_columns)} binary features from {len(df_binary)} samples")

        return df_binary, self.feature_metadata

    def get_feature_control_info(self, feature_name: str) -> Dict:
        if feature_name not in self.feature_metadata:
            return None

        metadata = self.feature_metadata[feature_name]

        if metadata['type'] == 'continuous':
            return {
                'control_type': 'slider',
                'label': feature_name,
                'min': metadata['min'],
                'max': metadata['max'],
                'default': metadata['mean'],
                'threshold': metadata['threshold'],
                'help': f"Binary threshold: {metadata['threshold']:.2f} (mean)"
            }
        elif metadata['type'] == 'categorical':
            return {
                'control_type': 'selectbox',
                'label': feature_name,
                'options': metadata['categories'],
                'default': metadata['categories'][0],
                'help': f"Categories: {', '.join(map(str, metadata['categories']))}"
            }
        elif metadata['type'] == 'categorical_dummy':
            return {
                'control_type': 'checkbox',
                'label': feature_name,
                'parent': metadata['parent'],
                'category': metadata['category'],
                'default': False,
                'help': f"Part of {metadata['parent']}"
            }
        elif metadata['type'] == 'binary':
            return {
                'control_type': 'checkbox',
                'label': feature_name,
                'default': False,
                'help': 'Binary feature (0 or 1)'
            }

        return None

    def create_instance_from_ui_values(self, ui_values: Dict) -> Dict[str, int]:
        instance = {}

        for feature in self.binary_columns:
            if feature not in self.feature_metadata:
                instance[feature] = 0
                continue

            metadata = self.feature_metadata[feature]

            if metadata['type'] == 'continuous':
                original_name = metadata['original_name']
                if original_name in ui_values:
                    value = ui_values[original_name]
                    instance[feature] = 1 if value > metadata['threshold'] else 0
                else:
                    instance[feature] = 0

            elif metadata['type'] == 'categorical_dummy':
                if feature in ui_values:
                    instance[feature] = 1 if ui_values[feature] else 0
                else:
                    instance[feature] = 0

            elif metadata['type'] == 'binary':
                if feature in ui_values:
                    instance[feature] = 1 if ui_values[feature] else 0
                else:
                    instance[feature] = 0

        return instance

    def get_original_feature_groups(self) -> Dict[str, List[str]]:
        groups = {}

        for feature in self.binary_columns:
            if feature not in self.feature_metadata:
                continue

            metadata = self.feature_metadata[feature]
            original_name = metadata.get('original_name', feature)

            if metadata['type'] == 'categorical_dummy':
                parent = metadata['parent']
                if parent not in groups:
                    groups[parent] = []
                groups[parent].append(feature)
            else:
                if original_name not in groups:
                    groups[original_name] = []
                groups[original_name].append(feature)

        return groups



st.set_page_config(
    page_title="ACR Classification Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'df_binary' not in st.session_state:
    st.session_state.df_binary = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {}
if 'feature_active' not in st.session_state:
    st.session_state.feature_active = {}
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'active_section' not in st.session_state:
    st.session_state.active_section = 'dataset'



def main():
    st.title("Association Rule Classification Explorer")
    st.markdown("Interactive tool for exploring association rules and predictions")

    with st.sidebar:
        st.header("Configuration")

        with st.expander("Dataset Upload", expanded=st.session_state.active_section=='dataset'):
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a dataset in CSV format"
            )

            if uploaded_file is not None:
                file_content = uploaded_file.read().decode('utf-8')
                delimiter = detect_csv_delimiter(file_content)
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, delimiter=delimiter)
                st.session_state.df_original = df

                st.success(f"Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")

                st.write("**Target Selection:**")
                target_col = st.selectbox(
                    "Select target column",
                    options=df.columns.tolist(),
                    index=len(df.columns)-1,
                    help="Choose the column you want to predict"
                )
                st.session_state.target_column = target_col

                if st.button("Next: Configure Parameters", use_container_width=True):
                    st.session_state.active_section = 'parameters'
                    st.rerun()

        with st.expander("Algorithm Parameters", expanded=st.session_state.active_section=='parameters'):
            with st.form("parameters_form"):
                st.write("**Association Rule Parameters:**")

                minsup = st.slider(
                    "Minimum Support",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                    help="Minimum frequency for itemsets (lower = more rules, slower)"
                )

                minconf = st.slider(
                    "Minimum Confidence",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.65,
                    step=0.05,
                    help="Minimum confidence for rules (lower = more rules)"
                )

                train_button = st.form_submit_button("Train Model", use_container_width=True)

                if train_button:
                    preprocess_progress = st.progress(0)

                    preprocessor = DataPreprocessor()

                    current_step = [0]
                    total_steps = 5

                    def preprocess_callback(message):
                        current_step[0] += 1
                        progress = min(current_step[0] / total_steps, 1.0)
                        preprocess_progress.progress(progress)

                    df_binary, metadata = preprocessor.preprocess_to_binary(
                        df, target_col, progress_callback=preprocess_callback
                    )

                    st.session_state.preprocessor = preprocessor
                    st.session_state.df_binary = df_binary

                    X = df_binary.drop(target_col, axis=1)
                    y = df_binary[target_col]

                    preprocess_progress.progress(1.0)

                    train_progress = st.progress(0)

                    train_step = [0]
                    estimated_steps = 10

                    def train_callback(message):
                        train_step[0] += 1
                        progress = min(train_step[0] / estimated_steps, 0.95)
                        train_progress.progress(progress)

                    start_time = time.time()
                    model = ACRClassifier(minsup=minsup, minconf=minconf)
                    model.fit(X, y, className=target_col, progress_callback=train_callback)
                    train_time = time.time() - start_time

                    st.session_state.model = model

                    features_in_rules = model.get_all_features_in_rules()

                    for feature in X.columns:
                        if feature not in st.session_state.feature_values:
                            st.session_state.feature_values[feature] = 0
                        if feature not in st.session_state.feature_active:
                            st.session_state.feature_active[feature] = feature in features_in_rules

                    train_progress.progress(1.0)

                    st.success(f"Model trained successfully. Generated {len(model.rules)} association rules in {train_time:.2f} seconds.")
                    st.rerun()

    if st.session_state.model is None:
        st.info("Upload a dataset and train a model to get started")

        st.markdown("""
        1. **Upload Dataset**: Upload a CSV file in the sidebar
        2. **Select Target**: Choose the column you want to predict
        3. **Set Parameters**: Adjust minimum support and confidence thresholds
        4. **Train Model**: Click 'Train Model' to generate association rules
        5. **Set Features**: Use the feature controls to set values
        6. **View Predictions**: See real-time predictions and matching rules
        """)
    else:
        model = st.session_state.model
        preprocessor = st.session_state.preprocessor

        features_in_rules = model.get_all_features_in_rules()

        feature_groups = preprocessor.get_original_feature_groups()

        st.header("Feature Controls")
        st.write(f"Features in Rules: {len(features_in_rules)}")

        for original_name, binary_features in sorted(feature_groups.items()):
            if not any(f in features_in_rules for f in binary_features):
                continue

            first_feature = binary_features[0]
            control_info = preprocessor.get_feature_control_info(first_feature)

            if control_info is None:
                continue

            col1, col2, col3 = st.columns([1, 2, 4])

            with col1:
                active_key = f"active_{original_name}"
                is_active = st.checkbox(
                    "",
                    value=st.session_state.feature_active.get(first_feature, False),
                    key=active_key,
                    label_visibility="collapsed"
                )

            with col2:
                st.write(f"**{original_name}**")

            with col3:
                for bf in binary_features:
                    st.session_state.feature_active[bf] = is_active

                if is_active:
                    if control_info['control_type'] == 'slider':
                        value = st.slider(
                            original_name,
                            min_value=float(control_info['min']),
                            max_value=float(control_info['max']),
                            value=float(control_info['default']),
                            key=f"slider_{original_name}",
                            help=control_info['help'],
                            label_visibility="collapsed"
                        )

                        binary_value = 1 if value > control_info['threshold'] else 0
                        st.session_state.feature_values[first_feature] = binary_value

                    elif control_info['control_type'] == 'selectbox':
                        selected_categories = []
                        cat_cols = st.columns(len(binary_features))
                        for idx, bf in enumerate(binary_features):
                            bf_info = preprocessor.get_feature_control_info(bf)
                            if bf_info and bf_info['control_type'] == 'checkbox':
                                with cat_cols[idx]:
                                    is_checked = st.checkbox(
                                        f"{bf_info['category']}",
                                        value=st.session_state.feature_values.get(bf, 0) == 1,
                                        key=f"cat_{bf}"
                                    )
                                    st.session_state.feature_values[bf] = 1 if is_checked else 0
                                    if is_checked:
                                        selected_categories.append(bf_info['category'])

                    elif control_info['control_type'] == 'checkbox':
                        value = st.checkbox(
                            "On/Off",
                            value=st.session_state.feature_values.get(first_feature, 0) == 1,
                            key=f"binary_{original_name}",
                            help=control_info['help']
                        )
                        st.session_state.feature_values[first_feature] = 1 if value else 0
                else:
                    for bf in binary_features:
                        st.session_state.feature_values[bf] = 0
                    st.write("(Disabled)")

            st.markdown("---")

        instance = {}
        for feature, value in st.session_state.feature_values.items():
            if st.session_state.feature_active.get(feature, False):
                instance[feature] = value
            else:
                instance[feature] = 0

        result = model.predict_single(instance)

        st.header("Prediction Results")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Predicted Class")
            prediction = result['prediction']

            if prediction == 1:
                st.metric("", f"Class: {prediction}")
            else:
                st.metric("", f"Class: {prediction}")

            st.subheader("Confidence Scores")
            for class_val, score in result['class_scores'].items():
                st.metric(
                    label=f"Class {class_val}",
                    value=f"{score:.3f}",
                    delta=None
                )

        with col2:
            st.subheader("Matching Rules")

            matching_rules = result['matching_rules']

            if len(matching_rules) == 0:
                st.warning("No rules matched this input. Using majority class as default.")
            else:
                st.success(f"{len(matching_rules)} rules fired for this prediction")

                for rule in matching_rules:
                    antecedent_items = [item.replace(':1', '') for item in rule.antecedent]
                    consequent = rule.consequent[0]
                    antecedent_str = ', '.join(antecedent_items)

                    rule_text = f"{antecedent_str} -> {consequent} (confidence: {rule.confidence:.3f}, support (X): {rule.antecedentSUP}, support (XY): {rule.antecedentNConsequentSUP})"
                    st.text(rule_text)

        st.markdown("---")
        st.header("Rules Explorer")

        col1, col2 = st.columns(2)

        with col1:
            filter_class = st.selectbox(
                "Filter by class",
                options=['All'] + [str(c) for c in model.classValues],
                index=0
            )

        with col2:
            min_conf_filter = st.slider(
                "Minimum confidence to display",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )

        filtered_rules = model.rules

        if filter_class != 'All':
            class_label = f"{model.className}:{filter_class}"
            filtered_rules = [r for r in filtered_rules if r.consequent[0] == class_label]

        filtered_rules = [r for r in filtered_rules if r.confidence >= min_conf_filter]

        filtered_rules = sorted(filtered_rules, key=lambda r: r.confidence, reverse=True)

        st.write(f"**Showing {len(filtered_rules)} rules** (Total: {len(model.rules)})")

        if len(filtered_rules) > 0:
            rules_data = []
            for rule in filtered_rules:
                antecedent_str = ' & '.join([item.replace(':1', '') for item in rule.antecedent])
                consequent_str = rule.consequent[0]

                rules_data.append({
                    'Antecedent': antecedent_str,
                    'Consequent': consequent_str,
                    'Support (X)': rule.antecedentSUP,
                    'Support (XY)': rule.antecedentNConsequentSUP,
                    'Confidence': f"{rule.confidence:.4f}"
                })

            rules_df = pd.DataFrame(rules_data)
            st.dataframe(rules_df, use_container_width=True, hide_index=True)

            csv = rules_df.to_csv(index=False)
            st.download_button(
                label="Download Rules as CSV",
                data=csv,
                file_name="association_rules.csv",
                mime="text/csv"
            )
        else:
            st.info("No rules match the current filters.")



if __name__ == "__main__":
    main()
