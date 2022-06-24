# 기본 라이브러리 & ML 라이브러리
import sys, os  # sys, os호출 라이브러리
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import openpyxl  # excel읽기/저장
import tkinter
from tkinter import filedialog


# Pyqt5 ui
from PyQt5 import uic  # PyQt5 내 uic호출
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QHeaderView)  # PyQt5 내 위젯 호출
                            #어플 실행,   윈도우창 실행,  파일불러오기,  메세지박스,  테이블 위젯
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets # plotly 라이브러리 호출
import plotly.express as px #online

# DB 라이브러리
import sqlite3 as sq
##################################################
# 초기 Setting
global Databasename, X_data_study, feature_selected
Databasename = 'test.db'
form_class=uic.loadUiType('v2-dB_System.ui')[0]  # form_class변수에 ui호출

class MyWindow(QMainWindow, form_class):  # MyWindow Class로 생성 후, QMainWindow, UI 상속받기
    def __init__(self):
        super().__init__()
        self.setupUi(self) # UI를 통해 윈도우화면 자동 생성
        self.setWindowTitle("DIC - Project 8조")  # 윈도우 이름 설정

        #plotly 그래프 연결
        self.browser = QtWebEngineWidgets.QWebEngineView()
        self.L3_graph_features.addWidget(self.browser)

        # Button Event함수 (self를 적어, ui와 연결)
        self.L0_btn_Excelfile.clicked.connect(self.L0_btn_Excelfileq)  # Layer0에서 파일 불러오기
        self.L1_btn_OpenFile.clicked.connect(self.L1_btn_OpenFileq)  # Layer1에서 파일 불러오기
        self.L1_btn_preprocessing.clicked.connect(self.L1_btn_preprocessingq)  # Layer1에서 data 전처리
        self.L2_btn_OK.clicked.connect(self.L2_btn_OKq)  # Layer2에서 OK 시, event 연결
        self.L2_btn_NG.clicked.connect(self.L2_btn_NGq)  # Layer2에서 NG 시, event 연결
        self.L3_btn_Analysis.clicked.connect(self.L3_btn_Analysisq)  # Layer3에서 Analysis 진행

    def L0_btn_Excelfileq(self): #Template 폴더 선택/다운
        root=tkinter.Tk()
        root.withdraw()
        path=filedialog.askdirectory(initialdir="/", title="폴더를 선택해 주세요")#parent=root, initialdir="./", title="폴더를 선택해 주세요")
        xl=openpyxl.load_workbook('template.xlsx')
        xl.save(path+'/'+'template.xlsx')

    def L1_btn_OpenFileq(self):
        try:
            fname=QFileDialog.getOpenFileName(self, 'Open file', './data')
            if fname[0]:
                self.data=pd.read_csv(fname[0], encoding='cp949')
                # 나중에 excel로 부를수 있게 바꾸기
                QMessageBox.about(self, 'OK', '업로드 완료')
            else:
                QMessageBox.about(self, 'Warning', '파일선택 안됨')
        except (KeyError, UnicodeDecodeError, NameError, pd.errors.EmptyDataError, pd.errors.ParserError):
            QMessageBox.about(self, 'warning', '잘못된 데이터 선택되었음')

    def L1_btn_preprocessingq(self):
        # 데이터 전처리2 [범주형데이터를 수치형으로 바꾸기]
        ###  data_feature 중에서 범주형 값 선정하기   자동으로 찾고 바꿔줘야함.
        data_feature=['Real - 1','압입대','상/하부 Set','상/하부 Set','Motor 형상','Motor 적층',
                      'Stator 열박음','Motor 고정방식','흡입 구조','지지 구조','Accum 업체','Muffler상부 높이']
        for i in data_feature:  # 범주형 data를 모두 numerical data로 변환
            self.data[i],_=self.data[i].factorize()
        self.data.astype('float')  # data type을 학습시키기 위해 float타입으로 변환
        rawdata = self.data.astype(dtype='int64') # data type을 학습시키기 위해 int64로 통일함.
        self.L1_pbar.setValue(10) # 진행률 10%

        # Data X, Y 인자 나누기
        self.X_data = rawdata.iloc[:, :31]  ### X인자 선정 (위에서 알아서 찾고 받아와야함.)
        self.Y_data = rawdata.iloc[:,31]    # Y인자 선정
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.X_data, self.Y_data,test_size=0.2, random_state=11) # X, Y data 분류하기
        self.L1_pbar.setValue(20) # 진행률 20%

        # Table 채우기 - X인자
        X_index_range= len(self.X_data)    # shape 구하기
        X_columns_range= len(self.X_data.columns)  # shape 구하기
        self.L2_tbl_Xdata.setRowCount(X_index_range)  # X인자 table row 갯수 정하기
        self.L2_tbl_Xdata.setColumnCount(X_columns_range)  # X인자 table column 갯수 정하기
        self.L2_tbl_Xdata.setHorizontalHeaderLabels(self.X_data.columns) # X인자 table column 값 채우기
        for row_index, row in enumerate(self.X_data.index):
            for col_index, column in enumerate(self.X_data.columns):
                value = self.X_data.loc[row_index][col_index]
                item = QTableWidgetItem(str(value))
                self.L2_tbl_Xdata.setItem(row_index, col_index, item)
        self.L1_pbar.setValue(60) # 진행률 60%

        # Table 채우기 - Y인자
        self.L2_tbl_Ydata.setRowCount(X_index_range)  # Y인자 table row 갯수 정하기
        self.L2_tbl_Ydata.setColumnCount(1)  # Y인자 table column 갯수 정하기
        # self.L2_tbl_Ydata.setHorizontalHeaderLabels('Noise') # Y인자 table column 값 채우기
        for row_index, row in enumerate(self.Y_data.index):
            value = self.Y_data.loc[row_index]
            item = QTableWidgetItem(str(value))
            self.L2_tbl_Ydata.setItem(row_index, 0, item)
        self.L1_pbar.setValue(90)  # 진행률 90%

        # shape 출력
        X_label_shape= f'({X_index_range}  X  {X_columns_range})'  # shape 구하기
        Y_label_shape= f'({X_index_range}  X    )'  # shape 구하기
        self.L2_lbl_Xlabel.setText(X_label_shape)
        self.L2_lbl_Ylabel.setText(Y_label_shape)
        self.L1_pbar.setValue(100)  # 진행률 100%

    def L2_btn_OKq(self):
        # Data 학습
        clf = RandomForestRegressor(random_state=531)  # RandomForest 사용
        clf.fit(self.X_train, self.y_train)   # 학습
        # clf_predict=clf.predict(self.X_test)
        testscore=round(clf.score(self.X_test,self.y_test),3) *100  # 학습 점수
        testscore_label=f'({testscore} %)'  #학습 점수
        self.L3_lbl_TestScore.setText(testscore_label) # 학습 점수

        # 그래프 그리기 - 상관관계 (Feature Importances)
        importances = clf.feature_importances_
        self.indices = np.argsort(importances)[::-1]
        fig = px.bar(self.X_data.columns[self.indices], y=importances[self.indices], title="Feature Importances")
        ### Xticks update 시키기
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))

        # Table 채우기 - Feature인자 + ComboBox 값넣기
        self.L3_tbl_Features.setRowCount(len(self.X_data.columns)) # Feature인자 table row 갯수 정하기
        self.L3_tbl_Features.setColumnCount(1)  # Feature인자 table column 갯수 정하기
        self.L3_tbl_Features.setHorizontalHeaderLabels(['Feature 상관도 순위'])
        for row_index, row in enumerate(self.X_data.columns):
            value = self.X_data.columns[self.indices][row_index]
            item = QTableWidgetItem(str(value))
            self.L3_tbl_Features.setItem(row_index, 0, item)
        feature_count = range(1,len(self.X_data.columns)+1) #comboBox 갯수 나열
        combo_count=list(map(str, list(feature_count))) #comboBox list화
        self.L3_cb_comboBox.addItems(combo_count) #comboBox list반영

    def L2_btn_NGq(self):
        print('NG')

    def L3_btn_Analysisq(self):
        # 상관성 높은 데이터로 학습 데이터 축소
        global X_data_study, feature_selected
        combo_selected= str(self.L3_cb_comboBox.currentText())
        feature_selected=[]

        for i in range(int(combo_selected)):
            feature_selected.append(self.L3_tbl_Features.item(i, 0).text())
        X_data_study=self.X_data[feature_selected]

        # DB 셋팅
        CreateTable()  #데이터베이스 생성

# DB 구성
def CreateTable():
    conn = sq.connect(Databasename)  # DB 연결
    # conn.execute("DELETE FROM users").rowcount
    cur = conn.cursor()
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='feature'" # DB 안에, table 속 결과 가져옴
    cur.execute(sql)
    rows = cur.fetchall()
    if not rows:  # DB table 없을 시, 생성
        sql="CREATE TABLE feature (F1 INTEGER, F2 INTEGER, F3 INTEGER, F4 INTEGER)"#F1 FLOAT, F2 FLOAT, F3 FLOAT, F4 FLOAT)"  idx INTEGER PRIMARY KEY,
        cur.execute(sql)
        conn.commit()
    conn.close()
    InsertData()

def InsertData():
    conn = sq.connect(Databasename)
    cur = conn.cursor()
    rows = cur.fetchall()
    if not rows:  # DB table 있을 시, 생성
        # sql = "TRUNCATE TABLE feature" # DB초기화 but sqlite에서는 TRUNCATE 명령어 지원안함.
        sql= "DELETE FROM feature"
        cur.execute(sql)
        conn.commit()
    for row in X_data_study.itertuples():
        sql = "insert into feature(F1, F2, F3, F4) values(?,?,?,?)"
        # cur.execute(sql, (X_data_study.iat[0, 0], X_data_study.iat[0, 1], X_data_study.iat[0, 2], X_data_study.iat[0, 3]))
        cur.execute(sql, (row[1], row[2], row[3], row[4]))
    conn.commit()
    conn.close()
    # SelectData()  # 데이터 입력 후 DB의 내용 불러와서 TableWidget에 넣기 위한 함수 호출

# def SelectData():  # DB 데이터 전체 선택
#     conn = sq.connect(Databasename)
#     cur = conn.cursor()
#     sql = "SELECT * FROM DIC-feature"
#     cur.execute(sql)
#     rows = cur.fetchall()
#     conn.close()
#     Real_Train_Data(rows) # DB의 내용을 불러와서 TableWidget에 넣기 위한 함수 호출
#
# def resource_path(relative_path):  #pyinstaller로 압축 시, db 위치전송
#     if hasattr(sys, '_MEIPASS'):
#         return os.path.join(sys._MEIPASS, relative_path)
#     return os.path.join(os.path.abspath("."), relative_path)

if __name__=="__main__":  # 프로그램 실행
    app=QApplication(sys.argv)  # QApplication 객체생성
    myWindow=MyWindow()   # MyWindow 함수 변수화
    myWindow.show()  # Window 화면 실행
    app.exec_()  # 이벤트 loop 실행

