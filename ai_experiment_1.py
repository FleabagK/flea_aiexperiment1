# ai_experiment_1.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import re
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

# 1. 加载数据
def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')

    X = train_df['text'].astype(str).tolist()
    y = train_df['target'].values
    X_test = test_df['text'].astype(str).tolist()

    return X, y, X_test



def clean_email(text):
    """去除部分邮件头字段，保留 Subject 和正文。"""
    fields_to_remove = [
        "From",
        "Organization",
        "Lines",
        "Reply-To",
        "NNTP-Posting-Host"
    ]

    for field in fields_to_remove:
        text = re.sub(
            rf"^{re.escape(field)}:.*\n?",
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE
        )

    return text
def clean_email_final(text):
    """最终方案：仅删除 NNTP-Posting-Host 字段。"""
    return re.sub(
        r"^NNTP-Posting-Host:.*\n?",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

X, y, X_test_unlabeled = load_data()

print("--- 数据加载成功 ---")
print(f"有标签数据: {len(X)}")
print(f"标签数量: {len(y)}")
print(f"无标签测试数据: {len(X_test_unlabeled)}")

# 2. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"训练集: {len(X_train)}")
print(f"验证集: {len(X_val)}")

# 3. TF-IDF 特征提取
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print(f"TF-IDF 特征数量: {X_train_tfidf.shape[1]}")

# 4. 实验一：SVM

# SVM：比较不同的 C 参数
for c in [0.1, 1, 10, 100]:
    svm_model = SVC(kernel='linear', C=c, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)

    svm_pred = svm_model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, svm_pred)
    f1 = f1_score(y_val, svm_pred, average='macro')

    print(f"\n--- SVM (C={c}) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")

# 5. 实验二：Naive Bayes

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_val_tfidf)


# Logistic Regression：比较不同的 C 参数

for c in [0.1, 1, 10, 100]:
    lr_model = LogisticRegression(
        C=c,
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train_tfidf, y_train)

    lr_pred = lr_model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, lr_pred)
    f1 = f1_score(y_val, lr_pred, average='macro')

    print(f"\n--- Logistic Regression (C={c}) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")


print("\n--- Naive Bayes ---")
print(f"Accuracy: {accuracy_score(y_val, nb_pred):.4f}")
print(f"Macro-F1: {f1_score(y_val, nb_pred, average='macro'):.4f}")


# 实验：比较原始文本与清洗邮件头后的分类效果
print("\n--- 文本清洗对照实验 ---")

# 清洗训练集和验证集，标签及数据划分保持不变
X_train_clean = [clean_email(text) for text in X_train]
X_val_clean = [clean_email(text) for text in X_val]

# 只在清洗后的训练文本上拟合 TF-IDF
clean_vectorizer = TfidfVectorizer(max_features=5000)
X_train_clean_tfidf = clean_vectorizer.fit_transform(X_train_clean)
X_val_clean_tfidf = clean_vectorizer.transform(X_val_clean)

# 使用逻辑回归，固定 C=100
clean_model = LogisticRegression(C=100, max_iter=1000, random_state=42)
clean_model.fit(X_train_clean_tfidf, y_train)
clean_pred = clean_model.predict(X_val_clean_tfidf)

print("清洗后文本：")
print(f"Accuracy: {accuracy_score(y_val, clean_pred):.4f}")
print(f"Macro-F1: {f1_score(y_val, clean_pred, average='macro'):.4f}")


# 实验：MLP 多层感知机 + 训练损失曲线
print("\n--- MLP ---")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=30,
    random_state=42
)

mlp_model.fit(X_train_tfidf, y_train)

mlp_pred = mlp_model.predict(X_val_tfidf)

print(f"Accuracy: {accuracy_score(y_val, mlp_pred):.4f}")
print(f"Macro-F1: {f1_score(y_val, mlp_pred, average='macro'):.4f}")
print(f"实际训练迭代次数: {mlp_model.n_iter_}")

# 绘制并保存训练损失曲线
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(mlp_model.loss_curve_) + 1),
    mlp_model.loss_curve_,
    marker='o'
)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("MLP Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_loss_curve.png", dpi=200)
plt.close()

print("损失曲线已保存：mlp_loss_curve.png")


# 8. 使用全部有标签数据训练最终模型，并预测测试集

print("\n--- 最终模型训练与测试集预测 ---")

# 1. 清洗全部有标签文本和无标签测试文本
X_clean = [clean_email_final(text) for text in X]
X_test_clean = [clean_email_final(text) for text in X_test_unlabeled]


# 2. 使用全部有标签文本拟合 TF-IDF
final_vectorizer = TfidfVectorizer(max_features=5000)
X_all_tfidf = final_vectorizer.fit_transform(X_clean)
X_test_tfidf = final_vectorizer.transform(X_test_clean)

# 3. 使用全部有标签数据训练逻辑回归模型
final_model = LogisticRegression(
    C=100,
    max_iter=1000,
    random_state=42
)
final_model.fit(X_all_tfidf, y)

# 4. 预测无标签测试数据
test_predictions = final_model.predict(X_test_tfidf)

# 5. 保存预测结果
submission = pd.DataFrame({"target": test_predictions})
submission.to_csv("predictions.csv", index=False)

print(f"最终训练数据量：{len(X)}")
print(f"测试集预测数量：{len(test_predictions)}")
print("预测文件已保存：predictions.csv")
print("\n预测标签分布：")
print(submission["target"].value_counts().sort_index())


# 9. 邮件头字段消融实验
print("\n--- 邮件头字段消融实验 ---")

fields_to_remove_list = [
    "From",
    "Organization",
    "Lines",
    "Reply-To",
    "NNTP-Posting-Host"
]

# 先记录原始文本基线
ablation_results = []

def evaluate_text_variant(remove_fields):
    """删除指定邮件头字段，使用相同设置评估逻辑回归。"""
    def process_text(text):
        for field in remove_fields:
            text = re.sub(
                rf"^{re.escape(field)}:.*\n?",
                "",
                text,
                flags=re.MULTILINE | re.IGNORECASE
            )
        return text

    train_text = [process_text(text) for text in X_train]
    val_text = [process_text(text) for text in X_val]

    vectorizer = TfidfVectorizer(max_features=5000)
    train_tfidf = vectorizer.fit_transform(train_text)
    val_tfidf = vectorizer.transform(val_text)

    model = LogisticRegression(
        C=100,
        max_iter=1000,
        random_state=42
    )
    model.fit(train_tfidf, y_train)
    pred = model.predict(val_tfidf)

    return (
        accuracy_score(y_val, pred),
        f1_score(y_val, pred, average="macro")
    )

# 原始文本基线
acc, f1 = evaluate_text_variant([])
ablation_results.append({
    "删除字段": "不删除（基线）",
    "Accuracy": acc,
    "Macro-F1": f1
})

# 每次只删除一个字段
for field in fields_to_remove_list:
    acc, f1 = evaluate_text_variant([field])
    ablation_results.append({
        "删除字段": field,
        "Accuracy": acc,
        "Macro-F1": f1
    })

# 删除全部五个字段
acc, f1 = evaluate_text_variant(fields_to_remove_list)
ablation_results.append({
    "删除字段": "全部五个字段",
    "Accuracy": acc,
    "Macro-F1": f1
})

ablation_df = pd.DataFrame(ablation_results)
print(ablation_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 保存实验结果，便于报告引用
ablation_df.to_csv("ablation_results.csv", index=False)
print("\n消融实验结果已保存：ablation_results.csv")

