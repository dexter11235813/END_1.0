# 1 write a python function to add to add two numbers


def return_exponential(num1, num2):
    return num1 ** num2


# 2 write a python function to split a string at space


def string_split_at_space(string):
    return string.split()


# 3 write a python program to convert a string to a char array


def char_array(string):
    return list(string)


# 4 write a python function to print the factorial of a number


def factorial(x):
    prod = 1
    for i in range(1, x + 1):
        prod *= i

    return prod


# 5 write a python function to accept a number and return all the numbers from 0 to that number


def print_numbers(x):
    for i in range(x):
        print(i)


# 6 write a python function that concatenates two stings


def concat(s1, s2):
    return s1 + s2


# 7 write a python function to return every second number from a list


def every_other_number(lst):
    return lst[::2]


# 7 write a python function to return every nth number from a list


def every_nth_number(lst, n):
    return lst[::n]


# 8 write a python function to accept a key, value pair and return a dictionary


def create_dictionary(key, value):
    return {str(key): value}


# 9 write a python function to update a dictionary with a new key, value pair


def update_dictionary(dict, key, value):
    dict[str(key)] = value
    return dict


# 10 write a python function to return the median of a list


def calc_median(arr):
    arr = sorted(arr)
    if len(arr) / 2 == 0:
        return arr[len(arr) / 2]
    else:
        return (arr[len(arr) // 2] + arr[(len(arr) - 1) // 2]) / 2


# 11 write a python function to return the length of an array plus 27


def return_length(arr):
    return len(arr) + 27


# 12  write a python function to return the third last element of an array


def return_last(arr):
    return arr[-3]


# 13  write a function to calculate the mean of an array


def calc_mean(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
    return sum / len(arr)


# 14 write a function to perform insertion sort on an arary


def sort_insertion(arr):
    for i in range(1, len(arr)):
        tmp = arr[i]

        j = i
        while (j > 0) & (tmp < arr[j - 1]):
            arr[j] = arr[j - 1]
            j = j - 1
        arr[j] = tmp
    return arr


# 15 write a function to implement a binary tree


class BinTree:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key


# 16 write a function to immplement insert in binary search tree


class BinaryTreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key


class Tree:
    def insert(self, root, key):
        if root is None:
            return BinaryTreeNode(key)
        else:
            if root.val == key:
                return root
            elif root.val < key:
                root.right = self.insert(root.right, key)
            else:
                root.left = self.insert(root.left, key)

        return root


# 17 write a function to initialize a linked list


class Cell:
    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None


# 18 write a function to create a linked list with given length and print the list after


class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


class LList:
    def __init__(self):
        self.head = None


def create_linked_list(*args):
    linked_list = LList()
    linked_list.head = Node(args[0])
    prev = linked_list.head

    for i in range(1, len(args)):
        entry = Node(args[i])
        prev.next = entry
        prev = entry
    return


# 20 write a function which returns the count of each token in a given sentence as a dictionary

from collections import Counter


def count_tokens(sent):
    sent = list(sent)
    return dict(Counter(sent))


# 21 write a function that removes all the punctuations from a string


import string


def remove_punct(s):
    return "".join(ch for ch in s if ch not in set(string.punctuation))


# 22 write a function that counts the sum of every element in the odd place in a list

from functools import reduce


def count_second(lst):
    return reduce(lambda x, y: x + y, lst[::2])


# 23 write a function that returns the square root of the third power of every number in a list


def comp_power(lst):
    return list(map(lambda x: x ** 1.5, lst))


# 23 write a function to calculate the residual sum of squares between two lists of the same size


def rss(lst1, lst2):
    diff = [lst1[x] - lst2[x] for x in range(len(lst1))]
    return sum(list(map(lambda x: x ** 2, diff)))


# 24 write a program to caclulate the approximate value of pi using the monte carlo method

import random


def pi_monte_carlo(n=1000000):
    count = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x ** 2 + y ** 2 <= 1:
            count += 1
    return 4 * count / n


print(pi_monte_carlo())

# 25 write a funtion to print all the files in the current directory

import os


def list_files():
    return os.listdir()


# 26 write a program to calculate the root of a nonlinear equation using Newton's method


class NewtonRaphsonSolver:
    def __init__(self, f, x, dfdx, min_tol=1e-3):
        self.func = f
        self.x = x
        self.derivative = dfdx
        self.min_tol = min_tol

    def calculate(self):
        func_val = self.func(self.x)
        iterations = 0
        while abs(func_val) > self.min_tol and iterations < 100:
            self.x = self.x - float(func_val) / self.derivative(self.x)
            func_val = self.func(self.x)
            iterations += 1

        if iterations <= 100:
            return self.x
        else:
            return None


def f(x):

    return x ** 4 - 16


def dfdx(x):
    return 4 * x ** 3


nrs = NewtonRaphsonSolver(f, 10, dfdx)
print(nrs.calculate())

# 26 write a generator in python which returns a random number between 0 and a million

import random


def yield_a_number():
    yield random.randint(0, 1000000)


# 27 write a program that filters a list for even numbers only and returns their sum


def map_reduce(lst):
    return reduce(lambda x, y: x + y, filter(lambda x: x % 2 == 0, lst))


print(map_reduce([1, 2, 3, 4, 5]))

# 28 write a program that return the first n numbers from a list


def sub_list(lst, ind):
    return lst[:ind]


print(sub_list([1, 2, 3, 4, 5, 56], 3))

# 29 write a program to sort a list using bubblesort


def bubblesort(arr):
    n = len(arr)

    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


print(bubblesort([1, 33, 192, 21, 0]))

# 30 write a function that accepts two numbers or lists or dictionaries and returns True if the two are equal, and False otherwise


def check_assert(item1, item2):
    try:
        assert item1 == item2
        return True
    except AssertionError:
        return False


# 31 write a function that checks if a number is an Armstrong number (sum of digits of the number = the number)

from itertools import chain


def check_armstrong(n):
    sum_of_digits = sum(map(lambda x: int(x) ** 3, chain(str(n))))
    if sum_of_digits == n:
        return True
    else:
        return False


# 32 write a program in python to create a directed graph, and add an edge between two vertices

from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, f, t):
        self.graph[f].append(t)

    def printEdge(self):
        for ed in list(self.graph.keys()):
            print(f"From : {ed}, To : {self.graph[ed]}")


g = Graph()
g.addEdge("a", "b")
g.addEdge("a", "e")
g.addEdge("b", "d")
g.addEdge("c", "d")
g.addEdge("c", "a")
g.printEdge()

# 33 write a program that shows how child class can access the init method of the parent class using super


class A:
    def __init__(self):
        print("My name is GYOBU MASATAKA ONIWA!")


class B(A):
    def __init__(self):
        super(B, self).__init__()
        print("as I breath, you WILL not pass the castle gates!")


tmp = B()

## 34 write a program to generate a random number between two ranges

import random


def rand_range(low, high):
    return random.randrange(low, high)


# 35 Write a python function that sorts a list of strings by their length in the descending order
def sort_by_len(arr):
    return sorted(arr, reverse=True, key=lambda x: len(x))


# 36 Write a  python function that returns the Highest Common Factor of two given numbers
def calculate_hcf(x1, x2):
    if x1 == 0:
        return x2
    else:
        return hcf(x2 % x1, x1)


# 37 Write a python program to calculate the LCM and HCF of two given numbers
def hcf(x1, x2):
    if x1 == 0:
        return x2
    else:
        return hcf(x2 % x1, x1)


def lcm_hcf(x1, x2):
    h_c_f = hcf(x1, x2)
    lcm = x1 * x2 / h_c_f
    return lcm, h_c_f


l, h = lcm_hcf(18, 12)

print(f"LCM : {l}, HCF: {h}")
# 38 write a python program which takes in a dictionary with unique values and converts keys into values and vice versa


def flip_dict(d):
    tmp_dict = {}
    for pair in d.items():
        tmp_dict[pair[1]] = pair[0]
    return tmp_dict


print(flip_dict({"a": 10, "b": 20, "c": 15}))


# 39 write a python function to return a list of all punctuations from the string library

import string


def return_punct():
    return string.punctuation


# 40 write a python function that takes in a string and returns it in lowercase


def to_lower(s):
    return s.lower()


# 41 write a python function that takes in a string and returns it in uppercase


def to_upper(s):
    return s.upper()


# 42 write a python program that converts lower case letters to uppercase and vice versa
def flip_case(s):
    s = [int(ord(x)) for x in s]
    s = [x - 32 if x >= 97 else x + 32 for x in s]
    s = [chr(x) for x in s]
    return "".join(s)


# 43 Define a function which returns the current working directory
import os


def get_cwd():
    return os.getcwd()


# 44 Define a python function that can read text file from a given URL
import requests


def read_data(url):
    data = requests.get(url).text
    return data


# 45 Define a python function which can generate a list where the values are square of numbers between 1 and 20 (both included). Then the function needs to print the last 5 elements in the list.

import requests


def get_status(url):
    data = requests.get(url)
    return data.status_code


# 46 Define a function which can generate a list where the values are square of numbers between 1 and 20 (both included). Then the function needs to print all values except the first 5 elements in the list.
import requests


def get_encoding(url):
    data = requests.get(url)
    return data.encoding


# 47 write a python function that accepts a valid path and changes the current working directory
import os


def change_dir(path):
    return os.chdir(path)


# 48 write a python function that checks if a given key is present in the environment
import os


def get_env_path(key):
    return os.getenv(key)


# 49 Write a generator that returns True / False randomly

import random


def generate_tf():
    rand = random.random()
    if rand > 0.5:
        yield True
    else:
        yield False


# 50 write a python program to normalize an array such that it sums upto 1


def normalize(arr):
    return [float(i) / sum(arr) for i in arr]


print(normalize([1, 2, 3, 4, 5]))


# 51 write a python program to perform Softmax operation on an input array

import math


def softmax(arr):
    e_arr = [math.exp(x) for x in arr]
    e_soft = [i / sum(e_arr) for i in e_arr]
    return e_soft


print(softmax([3.0, 1.0, 0.2]))


# 52 Write a python program to calculate the slope of a line given two points


def slope_of_a_line(x1, x2, y1, y2):
    del_x = x2 - x1
    del_y = y2 - y1
    return float(del_y) / del_x


print(slope_of_a_line(0, 10, 0, 10))

# 53 write a python function which checks if a number is a perfect square
import math


def is_perfect_square(num):
    sq_root = round(math.sqrt(num))
    if num == sq_root ** 2:
        return True
    else:
        return False


# 54 Write a python function that implements the ReLU function


def relu(arr):
    return [x if x > 0 else 0 for x in arr]


# 55 Write a python program that pads a given python list to a given length at the end and prints the modified list


def pad_arr_end(arr, pad_len):
    pad_arr = [0] * (pad_len - len(arr))
    return arr.extend(pad_arr)


tmp = [1, 2, 3, 4, 5]
pad_arr_end(tmp, 10)
print(tmp)

# 55 Write a python program that pads a given python list to a given length at the start and prints the modified list


def pad_arr_start(arr, pad_len):
    pad_arr = [0] * (pad_len - len(arr))
    pad_arr.extend(arr)
    return pad_arr


tmp = [1, 2, 3, 4, 5]
x = pad_arr_start(tmp, 10)
print(x)

# 56 write a python function to implement the sigmoid activation function

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# 57 write a python function to implement the tanh activation function

import math


def tanh(x):
    return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)


# 58 Write a python program that calculates and prints the area of an ellipse
import math


class Ellipse:
    def __init__(self, a, b):
        self.major_axis = b
        self.minor_axis = a

    def area(self):
        return math.pi * self.major_axis * self.minor_axis


ellipse = Ellipse(2, 10)
print(ellipse.area())


# 59 Write a python program that adds a time delay between a loop that prints numbers between 0 and 10

import time


def print_loop_with_delay(sec):
    for i in range(0, 10):
        time.sleep(sec)
        print(i)


# 60 Write a function to return the the unique tokens from a string


def unique_tokens(st):
    return set(st)


# 61 write a python function to return the standard deviation of a list of numbers

import math


def st_dev(arr):
    avg = sum(arr) / len(arr)
    ss_dev = sum([(x - avg) ** 2 for x in arr])
    return math.sqrt(ss_dev / (len(arr) - 1))


# 62 write a python function to return mode of the data

import statistics


def mode(arr):
    return statistics.mode(arr)


# 63 Write a python function which returns true if all the numbers in a list negative, else return False


def are_all_negative(arr):
    filt_arr = list(filter(lambda x: x < 0, arr))
    if len(filt_arr) == len(arr):
        return True
    else:
        return False


# 64 Write a python function that checks if all the numbers in a list sum upto 1. Returns False otherwise


def sum_upto_one(arr):
    arr_sum = sum(arr)
    try:
        assert float(arr_sum) == 1.0
        return True
    except AssertionError:
        return False


# 65 write a program to output a random even number between 0 and 10 inclusive using random module and list comprehension.
import random

print(random.choice([i for i in range(11) if i % 2 == 0]))

# 66 write a program to output a random number, which is divisible by 5 and 7, between 0 and 10 inclusive using random module and list comprehension.
import random

print(random.choice([i for i in range(201) if i % 5 == 0 and i % 7 == 0]))

# 67 write a program to generate a list with 5 random numbers between 100 and 200 inclusive.
import random

print(random.sample(range(100), 5))

# 68 write a program to randomly generate a list with 5 even numbers between 100 and 200 inclusive.
import random

print(random.sample([i for i in range(100, 201) if i % 2 == 0], 5))

# 69 write a program to randomly generate a list with 5 numbers, which are divisible by 5 and 7 , between 1 and 1000 inclusive.
import random

print(random.sample([i for i in range(1, 1001) if i % 5 == 0 and i % 7 == 0], 5))

# 70 write a program to randomly print a integer number between 7 and 15 inclusive.
import random

print(random.randrange(7, 16))

# 71 write a python function to count the length of the string


def len_str(st):
    return len(st)


# 72 write a program to print the running time of execution of "1+1" for 100 times.
from timeit import Timer

t = Timer("for i in range(100):1+1")
print(t.timeit())

# 73 write a program to shuffle and print the list [3,6,7,8].
from random import shuffle

li = [3, 6, 7, 8]
shuffle(li)
print(li)

# 74 write a program to shuffle and print the list [3,6,7,8].
from random import shuffle

li = [3, 6, 7, 8]
shuffle(li)
print(li)

# 75 write a program to generate all sentences where subject is in ["I", "You"] and verb is in ["Play", "Love"] and the object is in ["Hockey","Football"].
subjects = ["I", "You"]
verbs = ["Play", "Love"]
objects = ["Hockey", "Football"]
for i in range(len(subjects)):
    for j in range(len(verbs)):
        for k in range(len(objects)):
            sentence = "%s %s %s." % (subjects[i], verbs[j], objects[k])
            print(sentence)

# 76 Write a program to print the list after removing delete even numbers in [5,6,77,45,22,12,24].
li = [5, 6, 77, 45, 22, 12, 24]
li = [x for x in li if x % 2 != 0]
print(li)

# 77 By using list comprehension, write a program to print the list after removing delete numbers which are divisible by 5 and 7 in [12,24,35,70,88,120,155].
li = [12, 24, 35, 70, 88, 120, 155]
li = [x for x in li if x % 5 != 0 and x % 7 != 0]
print(li)

# 78 By using list comprehension, write a program to print the list after removing the 0th, 2nd, 4th,6th numbers in [12,24,35,70,88,120,155].
li = [12, 24, 35, 70, 88, 120, 155]
li = [x for (i, x) in enumerate(li) if i % 2 != 0]
print(li)

# 79 By using list comprehension, write a program generate a 3*5*8 3D array whose each element is 0.
array = [[[0 for col in range(8)] for col in range(5)] for row in range(3)]
print(array)

# 80 By using list comprehension, write a program to print the list after removing the 0th,4th,5th numbers in [12,24,35,70,88,120,155].
li = [12, 24, 35, 70, 88, 120, 155]
li = [x for (i, x) in enumerate(li) if i not in (0, 4, 5)]
print(li)

# 81 By using list comprehension, write a program to print the list after removing the value 24 in [12,24,35,24,88,120,155].
li = [12, 24, 35, 24, 88, 120, 155]
li = [x for x in li if x != 24]
print(li)

# 82 With two given lists [1,3,6,78,35,55] and [12,24,35,24,88,120,155], write a program to make a list whose elements are intersection of the above given lists.
set1 = set([1, 3, 6, 78, 35, 55])
set2 = set([12, 24, 35, 24, 88, 120, 155])
set1 &= set2
li = list(set1)
print(li)


# 83 With a given list [12,24,35,24,88,120,155,88,120,155], write a program to print this list after removing all duplicate values with original order reserved.
def removeDuplicate(li):
    newli = []
    seen = set()
    for item in li:
        if item not in seen:
            seen.add(item)
            newli.append(item)
    return newli


li = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
print(removeDuplicate(li))

# 84 Define a class Person and its two child classes: Male and Female. All classes have a method "getGender" which can print "Male" for Male class and "Female" for Female class.
class Person(object):
    def getGender(self):
        return "Unknown"


class Male(Person):
    def getGender(self):
        return "Male"


class Female(Person):
    def getGender(self):
        return "Female"


aMale = Male()
aFemale = Female()
print(aMale.getGender())
print(aFemale.getGender())

# 85 write a program which count and print the numbers of each character in a string
dic = {}
s = "JRR Tolkien"
for s in s:
    dic[s] = dic.get(s, 0) + 1
print("\n".join(["%s,%s" % (k, v) for k, v in dic.items()]))

# 86 write a program which accepts a string and counts the number of words in it
def num_of_words(st):
    return len(st.split())


# 87 write a function which accepts a string prints the characters that have even indexes.
def every_alternate_char(s):
    s = s[::2]
    return s


#  88 write a program which prints all permutations of [1,2,3]
import itertools

print(list(itertools.permutations([1, 2, 3])))

# 89 Write a program to solve a classic ancient Chinese puzzle:  We count 35 heads and 94 legs among the chickens and rabbits in a farm. How many rabbits and how many chickens do we have?
def solve(numheads, numlegs):
    ns = "No solutions!"
    for i in range(numheads + 1):
        j = numheads - i
        if 2 * i + 4 * j == numlegs:
            return i, j
    return ns, ns


# 90 Write a python function to round down a given decimal number

import math


def apply_ceil(num):
    return math.ceil(x)


# 91 Write a python function to round up a given decimal number

import math


def apply_floor(num):
    return math.floor(num)


# 92 Write a python function to round off a given decimal number


def apply_round(num):
    return round(num)


# 93 write a python function to find One's compliment of a number

import math


def OnesComplement(num):
    bits = int(math.floor(math.log(num) / math.log(2)) + 1)
    return ((1 << bits) - 1) ^ num


# 94 write a python function that takes in a decimal number and prints it's binary representation


def dec2bin(num):
    print(format(num, "b"))


# 95 write a python function that accepts a binary string and converts it into an equivalent decimal number


def bin2dec(num):
    return int(num, 2)


# 96 write a python function that takes a number and returns an array of the number duplicated n times


def duplicate_array(num, n):
    num = [num] * n
    return num


# 97 write a python function that accepts a number, and returns the nearest square number
import math


def nearest_square(n):
    upp = math.floor(math.sqrt(n))
    low = math.floor(math.sqrt(n))
    upp_diff = upp ** 2 - n
    low_diff = n - low ** 2

    if upp_diff > low_diff:
        return upp
    else:
        return low


# 98 write a python function that calculates the midpoint between two numbers


def midpoint(a, b):
    lar = b if b > a else a
    sm = a if b > a else b

    return float(lar + sm) / 2


# 99 write a  python function that accepts a string and reverses it


def reverse(st):
    return st[::-1]


# 100 write a python program that checks if a string is a pallindrome


def is_palindrome(st):
    st = st.lower()
    rev_st = st[::-1]
    try:
        assert rev_st == st
        return True
    except AssertionError:
        return False


st = "Nitin"
print(is_palindrome(st))
