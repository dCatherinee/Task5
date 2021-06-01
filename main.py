import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

sample_size = -1

def load_files():
    book_df = pd.read_excel('05_Зачисление.xlsx')
    global sample_size
    sample_size = len(book_df.index)
    return book_df

def part_2_scatterplot(data):
    #xdata = data['GRE Score'].tolist()
    #ydata = data['Chance of Admit '].tolist()

    # График GRE – Chance of Admit
    fig1, ax1 = plt.subplots()
    ax1.scatter(x = data['GRE Score'], y = data['Chance of Admit '], color = '#2700ff')

    plt.title('Диаграмма рассеивания')
    plt.xlabel('GRE Sroce')
    plt.ylabel('Chance of Admit')
    plt.show()

    # График TOEFL – Chance of Admit
    fig2, ax2 = plt.subplots()
    ax2.scatter(x = data['TOEFL Score'], y = data['Chance of Admit '], color = '#2700ff')

    plt.title('Диаграмма рассеивания')
    plt.xlabel('TOEFL Score')
    plt.ylabel('Chance of Admit')
    plt.show()

    # График University Rating – Chance of Admit
    fig3, ax3 = plt.subplots()
    ax3.scatter(x = data['University Rating'], y = data['Chance of Admit '], color = '#2700ff')

    plt.title('Диаграмма рассеивания')
    plt.xlabel('University Rating')
    plt.ylabel('Chance of Admit')
    plt.show()

    # График SOP – Chance of Admit
    fig4, ax4 = plt.subplots()
    ax4.scatter(x = data['SOP'], y = data['Chance of Admit '], color = '#2700ff')

    plt.title('Диаграмма рассеивания')
    plt.xlabel('SOP')
    plt.ylabel('Chance of Admit')
    plt.show()

    # График LOR – Chance of Admit
    fig5, ax5 = plt.subplots()
    ax5.scatter(x = data['LOR '], y = data['Chance of Admit '], color = '#2700ff')

    plt.title('Диаграмма рассеивания')
    plt.xlabel('LOR')
    plt.ylabel('Chance of Admit')
    plt.show()

    # График CGPA – Chance of Admit
    fig5, ax5 = plt.subplots()
    ax5.scatter(x = data['CGPA'], y = data['Chance of Admit '], color = '#2700ff')

    plt.title('Диаграмма рассеивания')
    plt.xlabel('CGPA')
    plt.ylabel('Chance of Admit')
    plt.show()


def part_3_linear_regression_from_all(data):
    x_train, x_test, y_train, y_test = train_test_split(data[['GRE Score','TOEFL Score','University Rating','SOP','LOR ', 'CGPA']], data['Chance of Admit '], test_size=0.2, random_state=0)
    model = LinearRegression().fit(x_train, y_train)
    predict = model.predict(x_test)

    r2 = r2_score(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    rss = ((y_test - predict)**2).sum()
    print('Результаты по всем переменным:', '\n')
    print('Коэффицент детерминации', r2)
    print('Средняя квадратическая ошибка', mse)
    print('Остаточная сумма квадратов', rss)


def part_3_linear_regression_for_3(data):
    x_train, x_test, y_train, y_test = train_test_split(data[['GRE Score','TOEFL Score','CGPA']], data['Chance of Admit '], test_size=0.2, random_state=0)
    model = LinearRegression().fit(x_train, y_train)
    predict = model.predict(x_test)

    r2 = r2_score(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    rss = ((y_test - predict)**2).sum()
    print('Результаты по переменным GRE Score, TOEFL Score, CGPA :', '\n')
    print('Коэффицент детерминации', r2)
    print('Средняя квадратическая ошибка', mse)
    print('Остаточная сумма квадратов', rss)


def part_3_linear_regression_for_GRE_Score(data):
    X = data['GRE Score'].values.reshape(-1, 1)
    y = data['Chance of Admit '].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression().fit(x_train, y_train)
    predict = model.predict(x_test)

    plt.scatter(x_test, y_test,  color='gray')
    plt.plot(x_test, predict, color='red', linewidth=3)
    plt.title('Линейная регрессия GRE Score и Chance of Admit')
    plt.show()

    r2 = r2_score(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    rss = ((y_test - predict)**2).sum()
    print('Результаты по переменной GRE Score:', '\n')
    print('Коэффицент детерминации', r2)
    print('Средняя квадратическая ошибка', mse)
    print('Остаточная сумма квадратов', rss)


def part_3_linear_regression_for_TOEFL_Score(data):
    X = data['TOEFL Score'].values.reshape(-1, 1)
    y = data['Chance of Admit '].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression().fit(x_train, y_train)
    predict = model.predict(x_test)

    plt.scatter(x_test, y_test,  color='gray')
    plt.plot(x_test, predict, color='green', linewidth=3)
    plt.title('Линейная регрессия TOEFL Score и Chance of Admit')
    plt.show()

    r2 = r2_score(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    rss = ((y_test - predict)**2).sum()
    print('Результаты по переменной TOEFL Score:', '\n')
    print('Коэффицент детерминации', r2)
    print('Средняя квадратическая ошибка', mse)
    print('Остаточная сумма квадратов', rss)


def part_3_linear_regression_for_CGPA(data):
    X = data['CGPA'].values.reshape(-1, 1)
    y = data['Chance of Admit '].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression().fit(x_train, y_train)
    predict = model.predict(x_test)

    plt.scatter(x_test, y_test,  color='gray')
    plt.plot(x_test, predict, color='blue', linewidth=3)
    plt.title('Линейная регрессия CGPA и Chance of Admit')
    plt.show()

    r2 = r2_score(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    rss = ((y_test - predict)**2).sum()
    print('Результаты по переменной CGPA:', '\n')
    print('Коэффицент детерминации', r2)
    print('Средняя квадратическая ошибка', mse)
    print('Остаточная сумма квадратов', rss)   


def part_3_polynomial_for_University_Rating(data):
    X = data['University Rating'].values.reshape(-1, 1)
    y = data['Chance of Admit '].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=2)
    x_poly = poly_reg.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_poly, y_train)
    x_test_poly = poly_reg.fit_transform(x_test)
    poly_pred = model.predict(x_test_poly)
    #print(poly_pred)

    r2 = r2_score(y_test, poly_pred)
    mse = mean_squared_error(y_test, poly_pred)
    rss = ((y_test - poly_pred)**2).sum()
    print('Результаты по переменной University Rating:', '\n')
    print('Коэффицент детерминации', r2)
    print('Средняя квадратическая ошибка', mse)
    print('Остаточная сумма квадратов', rss)

    plt.scatter(x_test, y_test, color = 'gray')
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x_test, poly_pred), key=sort_axis)
    x_test, poly_pred = zip(*sorted_zip)
    plt.plot(x_test, poly_pred, color='red')
    plt.title('Полиномиальная регрессия University Rating и Chance of Admit')
    plt.show()


def part_3_polynomial_for_SOP(data):
    X = data['SOP'].values.reshape(-1, 1)
    y = data['Chance of Admit '].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=2)
    x_poly = poly_reg.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_poly, y_train)
    x_test_poly = poly_reg.fit_transform(x_test)
    poly_pred = model.predict(x_test_poly)

    r2 = r2_score(y_test, poly_pred)
    mse = mean_squared_error(y_test, poly_pred)
    rss = ((y_test - poly_pred)**2).sum()
    print('Результаты по переменной SOP:', '\n')
    print('Коэффицент детерминации', r2)
    print('Средняя квадратическая ошибка', mse)
    print('Остаточная сумма квадратов', rss)

    plt.scatter(x_test, y_test, color = 'gray')
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x_test, poly_pred), key=sort_axis)
    x_test, poly_pred = zip(*sorted_zip)
    plt.plot(x_test, poly_pred, color='green')
    plt.title('Полиномиальная регрессия SOP и Chance of Admit')
    plt.show()


def part_3_polynomial_for_LOR(data):
    X = data['LOR '].values.reshape(-1, 1)
    y = data['Chance of Admit '].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=2)
    x_poly = poly_reg.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_poly, y_train)
    x_test_poly = poly_reg.fit_transform(x_test)
    poly_pred = model.predict(x_test_poly)

    r2 = r2_score(y_test, poly_pred)
    mse = mean_squared_error(y_test, poly_pred)
    rss = ((y_test - poly_pred)**2).sum()
    print('Результаты по переменной LOR:', '\n')
    print('Коэффицент детерминации', r2)
    print('Средняя квадратическая ошибка', mse)
    print('Остаточная сумма квадратов', rss)

    plt.scatter(x_test, y_test, color = 'gray')
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x_test, poly_pred), key=sort_axis)
    x_test, poly_pred = zip(*sorted_zip)
    plt.plot(x_test, poly_pred, color='green')
    plt.title('Полиномиальная регрессия LOR и Chance of Admit')
    plt.show()


def main():
    df = load_files()
    #part_2_scatterplot(df)
    #part_3_linear_regression_from_all(df)
    #part_3_linear_regression_for_3(df)
    #part_3_linear_regression_for_GRE_Score(df)
    #part_3_linear_regression_for_TOEFL_Score(df)
    #part_3_linear_regression_for_CGPA(df)
    #part_3_polynomial_for_University_Rating(df)
    #part_3_polynomial_for_SOP(df)
    #part_3_polynomial_for_LOR(df)



if __name__ == "__main__":
    main()
