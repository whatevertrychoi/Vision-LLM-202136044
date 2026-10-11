
### B. 기초 Python 문법 복습
#### $1)$ 데이터 형식 (Data Types)
name = "철수"
age = 20
height = 178.31754081

print(f"이름: {type(name)}, 나이: {type(age)}, 키: {type(height)}")
print()
print("이름:", name, end=", ")
print("나이: " + str(age) + "세", end=", ")
print(f"키(m): {height/100:.2f}m", end=", ")
print(f"키(cm):", str(height)[0:5] + "cm", end="\n\n")
print("성인 여부:", age > 19)
score_str = "0.8825"

# TODO: score_str를 백분율로 표현한 뒤 정수로 변환하여 출력
score_int = int(float(score_str) * 100)
print(f"점수 평균: {score_int}점")  # 예상 출력 "점수 평균: 88점"

#### $2)$ List
##### Constructing List
lst1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # length: 10, index: 0-9
lst2 = list((map(int, input("숫자 입력 (띄어쓰기로 구분): ").split())))

print(lst1)
print(lst2)
##### List Indexing
print(lst1[0])
# print(lst1[3])
# print(lst1[9])
# print(lst1[10])

# print(lst1[-1])
# print(lst1[-2])
# print(lst1[-9])
# print(lst1[-10])
##### Change Elements in a List
print(lst1)
lst1[0] = "X"
print(lst1)
lst1[-1] = "Y"
print(lst1)
print(lst1[0:1])

# print(lst1[0:9])

# print(lst1[0:10])

# print(lst1[0:len(lst1)])

# print(lst1[0:-1])

# print(lst1[:])

# print(lst1[:3])

# print(lst1[3:])

# print(lst1[0:5:2])

# print(lst1[::2])

# print(lst1[::-1])


# new_lst = lst1[:3] + lst2
# print(new_lst)
##### Nested Loop
for i in range(10):        # i = 0           # i = 0           # i = 0
    for j in range(10):    # i = 0, j = 0    # i = 0, j = 1    # i = 0, j = 2
        my_tuple = (i,j)   # (0,0)           # (0,1)           # (0,2)
        print(my_tuple)
    print("\n------\n")
##### List Comprehension
my_list1 = []

for i in range(10):
    my_list1.append(i)
    
print(my_list1)
test_lst1 = [0, 10, 100]
test_lst2 = [x * 100 for x in test_lst1]

print(test_lst2)
my_arr1 = []

for i in range(10):
    for j in range(10):
        my_arr1.append([i, j])


print(my_arr1)
my_arr2 = [[i, j] for i in range(10) for j in range(10)]

print(my_arr2)
my_arr3 = [[i, j] for i in range(10) for j in range(10) if i <= 1 and j <= 1]

print(my_arr3)
scores = [[45, 88, 92],
          [30, 75, 60],
          [95, 40, 50]]

# TODO: List Comprehension을 활용해 50점 이상인 값만 통과시켜 담으세요.
passed_scores = [score for score_list in scores for score in score_list if score >= 50]
print("합격 점수:", passed_scores)

#### $3)$ Enumerate & Zip
students = ["짱구", "철수", "맹구"]
scores = [85, 92, 78]
print(tuple(enumerate(students)))
for idx, name in enumerate(students):
    print(f"{idx}번 학생: {name}")
for idx, name in enumerate(students, start=1):
    print(f"{idx}번 학생: {name}")
print(tuple(zip(students, scores)))
for name, score in zip(students, scores):
    print(f"{name}의 점수: {score}점")
# TODO: enumerate와 zip을 조합하거나 활용하여 80점 이상 학생 정보 출력
for idx, (name, score) in enumerate(zip(students, scores), start=1):
    if score >= 80:
        print(f"{idx}번 {name} 학생 (점수: {score}점)")

# TODO: 예상 출력 "1번 짱구 학생 (점수: 85점)"

#### $4)$ Dictionary, JSON, YAML
##### Dictionary
student_info = {
    "name": "맹구",
    "age": 21,
    "major": "컴퓨터공학"
}

for key, value in student_info.items():
    print(f"{key}: {value}")
print("이름:", student_info["name"])

student_info["gpa"] = 3.8
student_info["languages"] = ["Python", "C++"]

print("학점:", student_info["gpa"])
print("프로그래밍 언어:", student_info["languages"])
##### JSON
import json
with open("맹구_개인정보.json", "w", encoding="utf-8") as f:
    json.dump(student_info, f, ensure_ascii=False, indent=4)

print("JSON 파일 저장 완료!")
##### YAML
import yaml
with open("맹구_개인정보.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(student_info, f, allow_unicode=True, default_flow_style=False)

print("YAML 파일 저장 완료!")
#### $5)$ NumPy
##### 기초 문법
import numpy as np
str_arr = np.array(["1.5", "2.8", "3.1"])
print(str_arr)

print()

float_arr = str_arr.astype(np.float64)
print(float_arr.dtype)
print(float_arr)
float_arr = np.array(["1.5", "2.8", "3.1"]).astype(np.float64)
print(float_arr.dtype)
print(float_arr)
##### Indexing 및 Slicing
grid = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("우측 상단 2x2 영역:\n", grid[:2, 1:])
##### NumPy 연산 함수
np_arr = np.array([1, 2, 3, 4, 5])

print("Sum:", np_arr.sum())
print("Minimum:", np_arr.min())
print("Maximum:", np_arr.max())
print("Mean:", np_arr.mean())
print("Argmax:", np_arr.argmax())
# [row: 학생, column: 과목]
scores = np.array([
    [83, 92, 100],  # 0번 학생
    [76, 63, 58],   # 1번 학생
    [91, 99, 84]    # 2번 학생
])

student_means = np.mean(scores, axis=1)
print("학생별 평균 (axis=1):")
print(student_means)

print()

subject_means = np.mean(scores, axis=0)
print("과목별 평균 (axis=0):")
print(subject_means)
scores = np.array([85, 90, 100, 60, 90, 75, 90])

is_ninety = (scores == 90)
print("90점인 점수 Boolean 배열:", is_ninety)
idx_ninety = np.where(scores == 90)
print(list(idx_ninety[0]))

print("90점인 학생 수:", (scores == 90).sum())
print("80점 이상인 학생 수:", (scores >= 80).sum())
# TODO: NumPy 연산 함수를 사용하여 배열을 정규화

np_arr = np.array([10, 20, 30, 40, 50])
np_arr_normalized = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())
print(np_arr_normalized)

# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0

np_arr = np.array([10, 20, 30, 40, 50], dtype=np.float32)
np_arr_final = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())
np_arr_final = np_arr_final.astype(np.float32)
print(np_arr_final)
print(np_arr_final.dtype)

##### Flatten
grid = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

flat_grid = grid.flatten()

print("Flatten 이전 모양 (shape):", grid.shape)
print("Flatten 이후 모양 (shape):", flat_grid.shape)
print("Flatten된 배열:", flat_grid)

## <center>< Section Project ></center>
raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
    ]

raw_scores = np.array([
    [95, 82, 88],   # 짱구
    [70, 65, 90],   # 철수
    [98, 100, 92],  # 맹구
    [60, 75, 80],   # 유리
    [81, 88, 84],   # 훈이
])

subjects = ["국어", "수학", "과학"]

# TODO: list comprehension을 활용해 raw_students 리스트에서 이름만 추출한 리스트 생성
student_names = [name.split("_")[1] for name in raw_students]
print("1. 학생 이름: ", end="")
for idx, name in enumerate(student_names):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(name, end=sep)

# TODO: 전산 시스템에 입력된 모든 성적들을 1차원 배열로 출력 (총 15개)
all_scores = raw_scores.flatten()
print("2. 전산 시스템에 입력된 모든 점수: ", end="")
for idx, score in enumerate(all_scores):
    sep = ", " if idx < len(all_scores) - 1 else "\n\n"
    print(f"{score}점", end=sep)

# TODO: 전체 15개 성적 중 85점 이상 개수
over_85 = np.count_nonzero(raw_scores >= 85)
print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")

# TODO: 학생별 평균 점수 출력
student_means = raw_scores.mean(axis=1)
mean_text = ", ".join(f"{student_names[i]} {student_means[i]:.2f}점" for i in range(len(student_names)))
print(f"4. 학생별 평균 점수: {mean_text}")

# TODO: 평균 점수가 가장 높은 학생 선택
top_student_idx = np.argmax(student_means)
top_student_name = student_names[top_student_idx]
print(f"5. 가장 높은 평균 점수 학생: {top_student_name} ({student_means[top_student_idx]:.2f}점)")

# TODO: 전체 점수 기준, 최고 점수와 과목 그리고 해당 학생 계산 (modulo % 사용 추천)
overall_max_score = raw_scores.max()
overall_max_idx = np.argmax(raw_scores)
overall_max_subject = subjects[overall_max_idx % len(subjects)]
overall_max_student = student_names[overall_max_idx // len(subjects)]
print(f"6. 전체 최고 점수: {overall_max_score}점 (과목: {overall_max_subject}, 학생: {overall_max_student})")
