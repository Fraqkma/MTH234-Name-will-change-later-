# Overfitting and Ridge Regression

Linear Algebra group project by a team of 6.

We use least squares from Section 6.5 of the textbook to show how a model that is too clever can fail on new data, and how ridge regression fixes it. Then we connect the result to machine learning.

**สรุปสั้นๆ ภาษาไทย** โปรเจกต์นี้ใช้กำลังสองน้อยที่สุดสร้างเส้นโค้งหลายระดับ แสดงให้เห็นว่าเส้นที่โค้งเกินไปจะ overfit และเดาข้อมูลใหม่พลาด แล้วแก้ด้วยริดจ์รีเกรสชัน ซึ่งเป็นการปรับสมการนอร์มอลเพียงเล็กน้อย

---

## The big idea

1. Least squares finds the curve that fits our data best.
2. If the curve has too many bends, it fits the noise too. This is called overfitting.
3. Ridge regression adds one small term to the normal equations and keeps the curve under control.

Ordinary least squares uses the normal equations.

```
A^T A x = A^T b
```

Ridge regression changes them to this.

```
(A^T A + lambda I) x = A^T b
```

When lambda is greater than 0, the matrix is always invertible, so the answer stays stable even when the columns of A are almost the same.

## Questions we answer

1. Is the curve that fits the old data most closely really the best curve
2. What causes overfitting and how can we see it
3. How does linear algebra fix it, and how does this connect to machine learning

## Team and work split

| Person | Topic | Hands in |
|---|---|---|
| 1 | Theory of least squares | Theory section, projection picture, 2 slides |
| 2 | Hand examples and QR method | Examples section, 2 slides |
| 3 | Data and models of many degrees | Shared code and data, curve plots |
| 4 | Show overfitting | MSE graph, condition number table |
| 5 | Ridge regression | Before and after graph, lambda graph |
| 6 | Machine learning link, conclusion, merging | ML section, introduction, conclusion, merged report and slides |

### Members

| Student ID | Name |
|---|---|
| 68070503421 | Nassasi Treesuptavee |
| 68070503425 | Thanatorn Charoensuk |
| 68070503434 | Nipitpon Choojit |
| 68070503436 | Parama Polsen |
| 68070503482 | Sitthawee Whangsawangwong |
| 68070503483 | Sirapobe Parinyarat |

Person 3 delivers the shared code and data first, because Person 4 and 5 build on them.

## Planned repository structure

```
.
├── README.md
├── requirements.txt
├── data/              shared data files made by Person 3
├── src/
│   ├── data.py        data creation and train and test split
│   ├── models.py      make_A, least squares solver, ridge
│   └── metrics.py     train_mse, test_mse, condition number
├── notebooks/         one notebook per person for experiments
├── figures/           all graphs and pictures
├── report/            report sections written by each person
└── slides/            presentation files
```

## Shared agreements

Everyone follows these so our work fits together.

- Use the shared data from the data folder. Do not create your own data.
- Use the same function names. These are `make_A`, `train_mse`, `test_mse` and `ridge`.
- Use the same random seed, which is 42, for all experiments.
- Make graphs in the same style. Use the same colors for training and test sets, and add axis labels and titles.
- Save every figure into the figures folder with a clear file name.
- Write the report in simple language.

## Experiment plan

| Step | What we do | Who |
|---|---|---|
| 1 | Create noisy data from a known curve such as sin of 2 pi x | Person 3 |
| 2 | Split into training and test sets, then fit degree 1 to 15 | Person 3 |
| 3 | Plot training and test MSE against degree and find where overfitting starts | Person 4 |
| 4 | Compute the condition number of A^T A at each degree | Person 4 |
| 5 | Apply ridge to a high degree model and try many values of lambda | Person 5 |
| 6 | Compare coefficients and test error before and after ridge | Person 5 |
| 7 | Link the results to machine learning and write the conclusion | Person 6 |

Theory and hand calculations by Person 1 and 2 run at the same time as steps 1 and 2.

## Getting started

Clone the repository and install the tools.

```bash
git clone <repository-url>
cd <repository-name>
pip install numpy matplotlib
```

Python with NumPy and Matplotlib is enough for the whole project. You can also open the notebooks in Google Colab if you do not want to install anything.

The core ridge function looks like this.

```python
import numpy as np

def ridge(A, b, lam):
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)
```

## Working with Git

- Do not push directly to the main branch. Make one branch for your own part, such as `person3-data`.
- Make small commits with clear messages.
- Open a pull request when your part is ready, and ask one teammate to check it.
- Person 6 merges the final report and slides.

## Key sentence for the presentation

The model that fits old data best is not always the best model. Ridge regression lets us give up a little accuracy on old data to guess new data better, and it comes from a small change to the normal equations that we learned in linear algebra.

## Reference

Section 6.5, Least Squares Problems, from the Linear Algebra textbook used in our course. This covers the normal equations, Theorem 14 on when the solution is unique, and Theorem 15 on the QR method.
