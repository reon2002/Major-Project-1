# major-project
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Iris major project 1.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyNS+jGcGzQlxpq9vL7xf0XP",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Aaaaaamz/Iris/blob/main/Iris_major_project_1.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XU-fbsTslTSD"
      },
      "source": [
        "**MAJOR PROJECT - 1 (COGNITIVE\n",
        "APPLICATION)**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LlVbVhG8l9AM"
      },
      "source": [
        "IRIS FLOWER CLASSIFICATION\n",
        "\n",
        "\n",
        "This is one of the most famous machine learning projects with Iris Flowers being the simplest\n",
        "machine learning datasets in classification literature. The dataset has numeric attributes and\n",
        "ML beginners need to figure out how to load and handle data. The iris dataset is small which\n",
        "easily fits into the memory and does not require any special transformations or scaling, to\n",
        "begin with.\n",
        "The goal of this machine learning project is to classify the flowers into among the three species – virginica,\n",
        "setosa, or versicolor based on length and width of petals and sepals."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "acI8Wl1FmFau"
      },
      "source": [
        "Importing the required libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SCUnq5j8c_fV"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WStGmprWSb8c"
      },
      "source": [
        "Importing the dataset and printint first values we can also use df tail to print last few values"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Qj0JA75YdqnQ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 203
        },
        "outputId": "637c4c5f-abaa-4282-a05f-f6034c1af3a8"
      },
      "source": [
        "df = pd.read_csv('iris.csv')\n",
        "df.head()"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>Sepal.Length</th>\n",
              "      <th>Sepal.Width</th>\n",
              "      <th>Petal.Length</th>\n",
              "      <th>Petal.Width</th>\n",
              "      <th>Species</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>5.1</td>\n",
              "      <td>3.5</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "      <td>setosa</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>4.9</td>\n",
              "      <td>3.0</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "      <td>setosa</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>4.7</td>\n",
              "      <td>3.2</td>\n",
              "      <td>1.3</td>\n",
              "      <td>0.2</td>\n",
              "      <td>setosa</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>4.6</td>\n",
              "      <td>3.1</td>\n",
              "      <td>1.5</td>\n",
              "      <td>0.2</td>\n",
              "      <td>setosa</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>5.0</td>\n",
              "      <td>3.6</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "      <td>setosa</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Unnamed: 0  Sepal.Length  Sepal.Width  Petal.Length  Petal.Width Species\n",
              "0           1           5.1          3.5           1.4          0.2  setosa\n",
              "1           2           4.9          3.0           1.4          0.2  setosa\n",
              "2           3           4.7          3.2           1.3          0.2  setosa\n",
              "3           4           4.6          3.1           1.5          0.2  setosa\n",
              "4           5           5.0          3.6           1.4          0.2  setosa"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Kumv5feOmwNM"
      },
      "source": [
        "checking the discription of data"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "mxmuSDFeaBzy",
        "outputId": "c33d5891-7cb3-4a91-90f4-1ae2953eae09"
      },
      "source": [
        "df.describe()"
      ],
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>Sepal.Length</th>\n",
              "      <th>Sepal.Width</th>\n",
              "      <th>Petal.Length</th>\n",
              "      <th>Petal.Width</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>150.000000</td>\n",
              "      <td>150.000000</td>\n",
              "      <td>150.000000</td>\n",
              "      <td>150.000000</td>\n",
              "      <td>150.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>75.500000</td>\n",
              "      <td>5.843333</td>\n",
              "      <td>3.057333</td>\n",
              "      <td>3.758000</td>\n",
              "      <td>1.199333</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>43.445368</td>\n",
              "      <td>0.828066</td>\n",
              "      <td>0.435866</td>\n",
              "      <td>1.765298</td>\n",
              "      <td>0.762238</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>4.300000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.100000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>38.250000</td>\n",
              "      <td>5.100000</td>\n",
              "      <td>2.800000</td>\n",
              "      <td>1.600000</td>\n",
              "      <td>0.300000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>75.500000</td>\n",
              "      <td>5.800000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>4.350000</td>\n",
              "      <td>1.300000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>112.750000</td>\n",
              "      <td>6.400000</td>\n",
              "      <td>3.300000</td>\n",
              "      <td>5.100000</td>\n",
              "      <td>1.800000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>150.000000</td>\n",
              "      <td>7.900000</td>\n",
              "      <td>4.400000</td>\n",
              "      <td>6.900000</td>\n",
              "      <td>2.500000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "       Unnamed: 0  Sepal.Length  Sepal.Width  Petal.Length  Petal.Width\n",
              "count  150.000000    150.000000   150.000000    150.000000   150.000000\n",
              "mean    75.500000      5.843333     3.057333      3.758000     1.199333\n",
              "std     43.445368      0.828066     0.435866      1.765298     0.762238\n",
              "min      1.000000      4.300000     2.000000      1.000000     0.100000\n",
              "25%     38.250000      5.100000     2.800000      1.600000     0.300000\n",
              "50%     75.500000      5.800000     3.000000      4.350000     1.300000\n",
              "75%    112.750000      6.400000     3.300000      5.100000     1.800000\n",
              "max    150.000000      7.900000     4.400000      6.900000     2.500000"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TCCiMUIgm5mC"
      },
      "source": [
        "checking which data type is used for which values"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BGNCGZRQarYW",
        "outputId": "ab293d39-bec5-45eb-a4a2-237364c52c96"
      },
      "source": [
        "df.info()"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 150 entries, 0 to 149\n",
            "Data columns (total 6 columns):\n",
            " #   Column        Non-Null Count  Dtype  \n",
            "---  ------        --------------  -----  \n",
            " 0   Unnamed: 0    150 non-null    int64  \n",
            " 1   Sepal.Length  150 non-null    float64\n",
            " 2   Sepal.Width   150 non-null    float64\n",
            " 3   Petal.Length  150 non-null    float64\n",
            " 4   Petal.Width   150 non-null    float64\n",
            " 5   Species       150 non-null    object \n",
            "dtypes: float64(4), int64(1), object(1)\n",
            "memory usage: 7.2+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WJ1YaTdXm__Y"
      },
      "source": [
        "From the above data we can se that width and length are of type float64 and sprcies is of object"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mntYB-W2nO1l"
      },
      "source": [
        " **displaing no. of samples on each class**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sgLa2n7vbKQw",
        "outputId": "a0121eaa-6b2b-4476-ba03-8e4de427f6fa"
      },
      "source": [
        "\n",
        "df['Species'].value_counts()"
      ],
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "virginica     50\n",
              "versicolor    50\n",
              "setosa        50\n",
              "Name: Species, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QhUJbyFancWE"
      },
      "source": [
        "preprocesssing the data and\n",
        "check for null values"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0na-U-vUbKzC",
        "outputId": "88c148ab-9866-4aaf-f7ac-8b5de6393826"
      },
      "source": [
        "df.isnull().sum()"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Unnamed: 0      0\n",
              "Sepal.Length    0\n",
              "Sepal.Width     0\n",
              "Petal.Length    0\n",
              "Petal.Width     0\n",
              "Species         0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "D-oA68N4nmyc"
      },
      "source": [
        "**plotting **\n",
        "\n",
        "Sepal.Length    \n",
        "Sepal.Width     \n",
        "Petal.Length    \n",
        "Petal.Width"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "6SWeuIusbywH",
        "outputId": "0433b684-6193-4a68-ad5c-8fc69e130db9"
      },
      "source": [
        "df['Sepal.Length'].hist()\n",
        "df['Sepal.Width'].hist()\n"
      ],
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f3a14f10190>"
            ]
          },
          "metadata": {},
          "execution_count": 39
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXAAAAD4CAYAAAD1jb0+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAASu0lEQVR4nO3dfYxldX3H8feXB2FlLEjAybCQLolmG8LGRSZUS2NmQAwW42JiGoklUG3GJkqwbtpS/9ndWhOa+tB/TFMUZZMiU8pDNIhWgoyUpGpncXWBdaPiatldWak8XULXLn77x5xth2GWe+bcc+fe3+X9Sm7mnjPn/O73t+fez5753fMQmYkkqTzHDLoASVIzBrgkFcoAl6RCGeCSVCgDXJIKddxqvthpp52W69ata7Tuc889x0knndRuQQM0Sv0Zpb6A/Rlmo9QXqN+fHTt2PJGZpy+dv6oBvm7dOubn5xutOzc3x9TUVLsFDdAo9WeU+gL2Z5iNUl+gfn8i4mfLzXcIRZIKZYBLUqEMcEkqlAEuSYUywCWpUAa4JBXKAJekQhngklQoA1ySCrWqZ2IK2Hryws/122DrphrLP93feiQVyz1wSSqUAS5JhTLAJalQBrgkFcoAl6RCGeCSVCgDXJIK1TXAI+LEiPhuRHw/Ih6OiG3V/Jsi4qcRsbN6bOx/uZKkI+qcyHMIuCgzOxFxPPBARHyt+t2fZ+Zt/StPknQ0XQM8MxPoVJPHV4/sZ1GSpO5iIZ+7LBRxLLADeD3w2cz8y4i4CXgLC3vo9wLXZeahZdadAWYAxsfHz5+dnW1UaKfTYWxsrNG6Q+XATgA6J5zB2KH93ZefGP6RqZHZNhX7M7xGqS9Qvz/T09M7MnNy6fxaAf5/C0ecAtwJXAP8F/AL4FXADcBPMvOvX279ycnJfMXflb66Fsrc+m1M7dlSY/nhvxbKyGybiv0ZXqPUF1jRXemXDfAVHYWSmU8B9wGXZuaBXHAI+CJwwUrakiT1ps5RKKdXe95ExBrgEuCHETFRzQvgcuChfhYqSXqxOkehTADbq3HwY4BbM/OuiPhmRJwOBLAT+NM+1ilJWqLOUSg/AM5bZv5FfalIklSLZ2JKUqEMcEkqlAEuSYUywCWpUAa4JBXKAJekQhngklQoA1ySCmWAS1KhDHBJKpQBLkmFMsAlqVAGuCQVygCXpEIZ4JJUKANckgpV5448GqTqJsj1lx/+myBLaod74JJUKANckgpV5670J0bEdyPi+xHxcERsq+afHRHfiYgfR8Q/R8Sr+l+uJOmIOnvgh4CLMvONwEbg0oh4M/C3wGcy8/XAk8AH+lemJGmprgGeCzrV5PHVI4GLgNuq+duBy/tSoSRpWZGZ3ReKOBbYAbwe+Czwd8C3q71vIuIs4GuZee4y684AMwDj4+Pnz87ONiq00+kwNjbWaN2hcmAnAJ0TzmDs0P7225/Y2H6bXYzMtqnYn+E1Sn2B+v2Znp7ekZmTS+fXOowwM18ANkbEKcCdwO/ULTAzbwBuAJicnMypqam6q77I3NwcTdcdKls3ATC3fhtTe7a03/4Vq38Y4chsm4r9GV6j1BfovT8rOgolM58C7gPeApwSEUf+AzgT2Ne4CknSitU5CuX0as+biFgDXALsZiHI31MtdhXw5X4VKUl6qTpDKBPA9moc/Bjg1sy8KyIeAWYj4m+A7wE39rFOSdISXQM8M38AnLfM/EeBC/pRlCSpO8/ElKRCGeCSVCgDXJIKZYBLUqEMcEkqlAEuSYUywCWpUAa4JBXKAJekQhngklQoA1ySClXreuCSmlt33VdbaWfv9Ze10o5Gh3vgklQoA1ySCmWAS1KhDHBJKpRfYkqvQH6xOhrcA5ekQhngklSoOnelPysi7ouIRyLi4Yi4tpq/NSL2RcTO6vEH/S9XknREnTHww8DmzHwwIl4D7IiIe6rffSYzP9m/8iRJR1PnrvQHgAPV82cjYjewtt+FSZJeXmRm/YUj1gH3A+cCHwWuBp4B5lnYS39ymXVmgBmA8fHx82dnZxsV2ul0GBsba7TuUDmwE4DOCWcwdmh/++1PbGy/zS5GZttU2u7Prn1Pt9LOhrUnN1pvuf4MuqamXqnvtenp6R2ZObl0fu0Aj4gx4FvAJzLzjogYB54AEvg4MJGZ73+5NiYnJ3N+fr7W6y01NzfH1NRUo3WHytaFN/zc+m1M7dnSh/bb+WCuxMhsm0rb/Rn0IXvL9WfQNTX1Sn2vRcSyAV7rKJSIOB64Hbg5M+8AyMzHM/OFzPwN8DnggpUULknqTZ2jUAK4EdidmZ9eNH9i0WLvBh5qvzxJ0tHUOQrlQuBKYFdE7KzmfQy4IiI2sjCEshf4YF8qlCQtq85RKA8Ascyv7m6/nMJsXd0vcCRpMc/ElKRCGeCSVCgDXJIKZYBLUqEMcEkqlDd00Mho6+zCzRsOM9VKS1J/uQcuSYUywCWpUAa4JBXKAJekQhngklQoj0JZzGubSCqIe+CSVCgDXJIKZYBLUqEMcEkqlAEuSYUywCWpUAa4JBWqzl3pz4qI+yLikYh4OCKureafGhH3RMSPqp+v7X+5kqQj6uyBHwY2Z+Y5wJuBD0XEOcB1wL2Z+Qbg3mpakrRKugZ4Zh7IzAer588Cu4G1wCZge7XYduDyfhUpSXqpyMz6C0esA+4HzgV+npmnVPMDePLI9JJ1ZoAZgPHx8fNnZ2cbFdrpdBgbG2u0bm0Hdva3/UU6J5zB2KH97Tc8sbH9NrtYlW1Tw659T7fSzvgaeN2p7V1Woa26NqxtVtNy22fQNTU1LO+1ttTtz/T09I7MnFw6v3aAR8QY8C3gE5l5R0Q8tTiwI+LJzHzZcfDJycmcn5+v9XpLzc3NMTU11Wjd2lbxWihz67cxtWdL+w1vbeeDuRKrsm1qaPOOPNe8b1MrbUF7de29/rJG6y23fQZdU1PD8l5rS93+RMSyAV7rKJSIOB64Hbg5M++oZj8eERPV7yeAg3WLliT1rs5RKAHcCOzOzE8v+tVXgKuq51cBX26/PEnS0dS5nOyFwJXArog4Mkj8MeB64NaI+ADwM+AP+1OiJGk5XQM8Mx8A4ii/vrjdciRJdXkmpiQVygCXpEIZ4JJUKANckgplgEtSobwrvaTG2jqjE1b/rM5R4B64JBXKAJekQhngklQoA1ySCmWAS1KhDHBJKpQBLkmFMsAlqVAGuCQVygCXpEJ5Kr1UiKanrW/ecJirWzzlXcPDPXBJKpQBLkmFqnNX+i9ExMGIeGjRvK0RsS8idlaPP+hvmZKkpersgd8EXLrM/M9k5sbqcXe7ZUmSuuka4Jl5P/CrVahFkrQCkZndF4pYB9yVmedW01uBq4FngHlgc2Y+eZR1Z4AZgPHx8fNnZ2cbFdrpdBgbG2u0bm0Hdva3/UU6J5zB2KH97Tc8sbH9NrtYlW1Tw659T7fSzvgaeN2pJ7fSFrRXV1Pja+Dx5wdaQi0b1nb/Nx+W91pb6vZnenp6R2ZOLp3fNMDHgSeABD4OTGTm+7u1Mzk5mfPz811fbzlzc3NMTU01Wre2re19aLuZW7+NqT1b2m946+qHxapsmxraujvM5g2HueZ9m1ppC9q9a00Tmzcc5lO7hv+I4Tp35BmW91pb6vYnIpYN8EZHoWTm45n5Qmb+BvgccEGTdiRJzTUK8IiYWDT5buChoy0rSeqPrn9XRcQtwBRwWkQ8BmwBpiJiIwtDKHuBD/axRknSMroGeGZesczsG/tQizQ0Bj1uLdXhmZiSVCgDXJIKZYBLUqEMcEkqlAEuSYUywCWpUAa4JBXKAJekQhngklQoA1ySCmWAS1KhDHBJKpQBLkmFMsAlqVAGuCQVygCXpEIN/51O1X8rvZnzAG6cLOml3AOXpEIZ4JJUqK4BHhFfiIiDEfHQonmnRsQ9EfGj6udr+1umJGmpOnvgNwGXLpl3HXBvZr4BuLealiStoq4Bnpn3A79aMnsTsL16vh24vOW6JEldRGZ2XyhiHXBXZp5bTT+VmadUzwN48sj0MuvOADMA4+Pj58/OzjYqtNPpMDY21mjd2g7s7G/7i3ROOIOxQ/tX7fVaNbHxRZOrsm1q2LWvnaNjxtfA48+30tRQKKU/G9Z2PxpqWN5rbanbn+np6R2ZObl0fs+HEWZmRsRR/xfIzBuAGwAmJydzamqq0evMzc3RdN3atm7qb/uLzK3fxtSeLav2eq264sVBuSrbpoarr/tqK+1s3nCYT+0anSNsS+nP3vdNdV1mWN5rbem1P02PQnk8IiYAqp8HG1cgSWqkaYB/Bbiqen4V8OV2ypEk1VXnMMJbgH8H1kfEYxHxAeB64JKI+BHwtmpakrSKug6MZeYVR/nVxS3XoleodS2NXUuvNJ6JKUmFMsAlqVAGuCQVygCXpEIZ4JJUKANckgplgEtSoQxwSSqUAS5JhTLAJalQBrgkFWr4LxLci63dLxAvSaVyD1ySCmWAS1KhDHBJKpQBLkmFGu0vMSUVo86NPTZvONz15tV7r7+srZJavdlIm3Ud4R64JBXKAJekQvU0hBIRe4FngReAw5k52UZRkqTu2hgDn87MJ1poR5K0Ag6hSFKhIjObrxzxU+BJIIF/zMwblllmBpgBGB8fP392drbRa3U6HcbGxla20oGdjV5rNXROOIOxQ/sHXUYzExtfNNlo2yyya9/TvVbUqvE18Pjzg66iPaPUn5L7smHtSy/tUfezMz09vWO5IepeA3xtZu6LiNcB9wDXZOb9R1t+cnIy5+fnG73W3NwcU1NTK1tpiK+FMrd+G1N7tgy6jGa2vjhwG22bRdo8VKsNmzcc5lO7RucI21HqT8l9We4wwrqfnYhYNsB7GkLJzH3Vz4PAncAFvbQnSaqvcYBHxEkR8Zojz4G3Aw+1VZgk6eX18rfIOHBnRBxp50uZ+fVWqpIkddU4wDPzUeCNLdby8g7shK2bVu3lJGnYeRihJBXKAJekQhngklQoA1ySCmWAS1KhDHBJKpQBLkmFMsAlqVAGuCQVygCXpEIZ4JJUqDIvrKvBWnqd9fXberpOzd4TXzpv3X9/qXF70iuFe+CSVCgDXJIKZYBLUqEMcEkqlAEuSYUywCWpUAa4JBWqpwCPiEsjYk9E/DgirmurKElSd40DPCKOBT4LvAM4B7giIs5pqzBJ0svrZQ/8AuDHmfloZv4amAW8bbwkrZLIzGYrRrwHuDQz/6SavhL43cz88JLlZoCZanI9sKdhracBTzRcdxiNUn9GqS9gf4bZKPUF6vfntzPz9KUz+34tlMy8Abih13YiYj4zJ1soaSiMUn9GqS9gf4bZKPUFeu9PL0Mo+4CzFk2fWc2TJK2CXgL8P4A3RMTZEfEq4L3AV9opS5LUTeMhlMw8HBEfBv4VOBb4QmY+3FplL9XzMMyQGaX+jFJfwP4Ms1HqC/TYn8ZfYkqSBsszMSWpUAa4JBVq6AM8Is6KiPsi4pGIeDgirh10TU1FxIkR8d2I+H7Vl22DrqkNEXFsRHwvIu4adC29ioi9EbErInZGxPyg6+lFRJwSEbdFxA8jYndEvGXQNTUVEeurbXLk8UxEfGTQdTUVEX9WZcBDEXFLRCxzY8Ea7Qz7GHhETAATmflgRLwG2AFcnpmPDLi0FYuIAE7KzE5EHA88AFybmd8ecGk9iYiPApPAb2XmOwddTy8iYi8wmZnFnywSEduBf8vMz1dHir06M58adF29qi7jsY+FEwd/Nuh6Vioi1rLw2T8nM5+PiFuBuzPzppW2NfR74Jl5IDMfrJ4/C+wG1g62qmZyQaeaPL56DPf/oF1ExJnAZcDnB12L/l9EnAy8FbgRIDN/PQrhXbkY+EmJ4b3IccCaiDgOeDWwv0kjQx/gi0XEOuA84DuDraS5arhhJ3AQuCczi+1L5e+BvwB+M+hCWpLANyJiR3UZiFKdDfwS+GI1vPX5iDhp0EW15L3ALYMuoqnM3Ad8Evg5cAB4OjO/0aStYgI8IsaA24GPZOYzg66nqcx8ITM3snDm6gURce6ga2oqIt4JHMzMHYOupUW/n5lvYuEqmx+KiLcOuqCGjgPeBPxDZp4HPAcUf8nnaijoXcC/DLqWpiLitSxc+O9s4AzgpIj4oyZtFRHg1Xjx7cDNmXnHoOtpQ/Xn7H3ApYOupQcXAu+qxo1ngYsi4p8GW1Jvqr0jMvMgcCcLV90s0WPAY4v+wruNhUAv3TuABzPz8UEX0oO3AT/NzF9m5v8AdwC/16ShoQ/w6ou/G4HdmfnpQdfTi4g4PSJOqZ6vAS4BfjjYqprLzL/KzDMzcx0Lf9Z+MzMb7UkMg4g4qfqinGq44e3AQ4OtqpnM/AXwnxGxvpp1MVDcF//LuIKCh08qPwfeHBGvrvLtYha+21uxvl+NsAUXAlcCu6qxY4CPZebdA6ypqQlge/Ut+jHArZlZ/KF3I2QcuHPhM8VxwJcy8+uDLakn1wA3V8MOjwJ/POB6elL9p3oJ8MFB19KLzPxORNwGPAgcBr5Hw1Pqh/4wQknS8oZ+CEWStDwDXJIKZYBLUqEMcEkqlAEuSYUywCWpUAa4JBXqfwFLFJtLiGD/vgAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "T5IxO3VWcgrm",
        "outputId": "b9bfd3c4-3eb1-45b1-a87f-e222ab9292ad"
      },
      "source": [
        "df['Petal.Width'].hist()"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f3a143d4390>"
            ]
          },
          "metadata": {},
          "execution_count": 49
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "EuaqHHE6dStW",
        "outputId": "7266a6c3-ea69-4980-df8d-d18d2fec46e8"
      },
      "source": [
        "df['Petal.Length'].hist()"
      ],
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f3a143c1450>"
            ]
          },
          "metadata": {},
          "execution_count": 50
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Owm7oThyn7N6"
      },
      "source": [
        "plotting sepal length and width using scatterplot"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 299
        },
        "id": "uUlmsR3heOet",
        "outputId": "45245a33-2eb8-4a9a-834e-50044e4eef45"
      },
      "source": [
        "colors = ['red', 'orange', 'blue']\n",
        "species = ['virginica','versicolor','setosa']\n",
        "for i in range(3):\n",
        "    x = df[df['Species'] == species[i]]\n",
        "    plt.scatter(x['Sepal.Length'], x['Sepal.Width'], c = colors[i], label=species[i])\n",
        "plt.xlabel(\"Sepal Length\")\n",
        "plt.ylabel(\"Sepal Width\")\n",
        "plt.legend()"
      ],
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7f3a142bddd0>"
            ]
          },
          "metadata": {},
          "execution_count": 53
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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hLt6BPcrKzeF9U033cTxxx/P8OHDuf766wFYtWoVxxxzDBdffDFDhw7lpZdeYsqUKQwdOpRhw4Zx8803A9HTVG/bto2pU6cybNgwRo0axUMPPbRXXK+++iof/ehHGT58OCeddBJPPvnk7vguuugiTjnlFC666KJuf39QYyBVUsEpZqSWrWyBBdNgy2rAg/sF07rdIEyaNGn3XEMA9913H3379uX5559nwYIFLFmyhEWLFvHwww8D8Pzzz9Pc3MxTTz3Fhg0bWLt2LcuXL2fZsmVMnTq1Q91vv/02kyZN4nvf+x5Lly5l3rx59O7dm1tuuQUzY9myZdx9991ccskley1kf/311zNq1CiefPJJvvGNb3DxxRfvfm3FihXMmzePu+++u1vfvY0aA6kKLUwjXbL0WmgtuL7YuiUo74b2U1gvXbqUPn36sGzZMn73u98xatQojjvuOJ555hmef/55AAYMGMBJJ50EwJFHHsnf/vY3Lr/8cn7zm9/snoa6TWfTVDc2NvLII49w4YUXAvC+972PAQMG7DVV9SOPPLL7P/+xY8eyceNG3njjDQAmTJhA7969u/W921OfgVRFW79AlaeYkbzZUuQ6YrHyMhROYb169WquueYaPve5z3XYbtWqVR0mjuvTpw9Lly7lt7/9LT/84Q+57777uO2227odT5SkJ6/TmYFUzeTJsGoV7NoV3KshkEhNRa4jFisvQ+EU1meddRa33XYbb775JgBr167llVde2et9GzZsYNeuXXz84x/n61//OosXL+7werFpqk899VRawuFzzz33HC+++CLHHHNMh/e232b+/Pkceuihe515JEVnBtKpDEwMKbK3ETcGfQTtLxU1NAXl3VQ4hXW/fv14+umnOfnkkwHYf//9mTNnDg0NDR3et3btWqZOncquXbsA+OY3v9nh9fbTVG/dupXevXszb948mpubmT59OsOGDaOxsZE77rijw2I4EHQUf+pTn2L48OE0NTVx5513dvt7FqMprGUvbTkA7Yd+NjXBrFlqECR5ZU9hvbIl6CPY8mJwRjDiRhikA7M9TWEtiVAOgGTaoMn6418B6jOQvSgHQKT+qDGQvSgHQKT+qDGQvSgHQKT+qDGQvUyeHHQWDxgAZsG9Oo9FalvFO5DNrAFYCKx193MLXtsX+AkwGtgITHL3VZWOSaJlYJlZEUlRGmcG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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "H6y2PXvtoFx5"
      },
      "source": [
        "similarly plotting petal length and petal width"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "V69_bJj9epVG",
        "outputId": "e172f2d0-b987-4fcb-fcff-8ed3b32af172"
      },
      "source": [
        "for i in range(3):\n",
        "    x = df[df['Species'] == species[i]]\n",
        "    plt.scatter(x['Petal.Length'], x['Petal.Width'], c = colors[i], label=species[i])\n",
        "plt.xlabel(\"Petal Length\")\n",
        "plt.ylabel(\"Petal Width\")\n",
        "plt.legend()\n",
        "\n"
      ],
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7f3a1474e190>"
            ]
          },
          "metadata": {},
          "execution_count": 57
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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mC2bordiFcjlowgxVl9BWf3IxG2hkPY8zJo1iy996Uts3bqVRx99lDVr1nDNNdcwe/ZsDjnkEDZv3szA5L+QgK1bt1JXV8dnP/tZxowZs/+v/G49p5o+4YQT2LFjB4MHD+aUU06htbWVM844g5deeonXXnuNMWPG8NRTT+3ft3uba665ht/97nccddRRfa44RLKWboGXXNvyPXeu22W7balEHGP1XRFMnA81SZdZNfVBex6Sp6GeMWMG559/PieddBJNTU3MnDmTHTt29Nlv8+bNnH766UyaNIkLLriA7373u73eTzfV9L59+2hqamLWrFksXry410I2EHQKr1ixggkTJnDVVVdx66235vUZRYoqipqBOIlT3KmGE8X1UZDho6/c4f7zBvdWC55fidGY4ZjQ8FEpqShqBuKkBHGTZvhoda5HIP3S71RKqrExfLrrhgbouXZ3ptvFTQniTrceQfXdGhKR+IuiZiBOYha3EoGIxE8UNQNxErO4lQhEJH6iqBmIk5jFrUQgIvETRc1AnMQsbnUWSyj9TkUqizqLY2bx4sVs2bKl1GGIFF+qsfOFXs8gTmP04xhPslTjSuP6qIRpqE877TRfvnx5qcNIq9x+p1IGUo2dnzs39zUOyqG2ICbxoPUIervjjmAqb7PguRD/PXbu3OnnnHOOT5gwwceNG+d33323t7W1+amnnupTpkzxGTNm+JYtW/y+++7zgw8+2EePHu0TJ070zs5OX7p0qU+aNMnHjx/vc+bM8V27drm7+9e//nU/7rjjvKmpyb/yla+4u/uDDz7o06ZN80mTJvn06dP9j3/8Y/7Bh1AikIJLNa9+TU14e67rGcRtjYGYxKNE0ENUyXnJkiX+xS9+cf/rt99+20866SR/88033d397rvv9jlz5rh77yuCd99910eMGOHr1693d/cLL7zQb7jhBt+6dauPHj3a9+3b5+7uf/rTn9zd/a233trf9pOf/MS//OUv5xd4CkoEUnBm+X3hp3qYZXae5O1K/bmLHE+6RFB1fQTz5kFn0izUnZ1Bez6ampp4+OGH+frXv87jjz/O66+/zvPPP89ZZ53FpEmT+Nd//Vfa29v77Ld+/XpGjRrF6NGjAbj44ot57LHHOOywwxg0aBCXXnopP/vZz6hPDDVrb2/nr//6r2lqauL6669n7dq1+QUuUiypxsjX1BT2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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "Ypf57xxueWhu",
        "outputId": "82280af2-7c8b-4c29-bfef-3aee595ca4ea"
      },
      "source": [
        "for i in range(3):\n",
        "    x = df[df['Species'] == species[i]]\n",
        "    plt.scatter(x['Sepal.Length'], x['Petal.Length'], c = colors[i], label=species[i])\n",
        "plt.xlabel(\"Sepal Length\")\n",
        "plt.ylabel(\"Petal Length\")\n",
        "plt.legend()"
      ],
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7f3a146fcb50>"
            ]
          },
          "metadata": {},
          "execution_count": 56
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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sncl/0ob3N8CPxgQ/HwlmamGb7rpJk488UQmTZrEddddB8CGDRsYO3Ysl112GRMmTOC1117jiiuuYMKECUycOJGbb74ZCJ9OePfu3cybN4+JEycydepUHn/88V5xbdu2jY997GNMmjSJmTNn8txzzx2I79JLL2XWrFlceuml/frsIlWpRtZWDZXSwiPZusx7pQ2Wz4d9ub+OO18NngOMKe8v40UXXcTVV1/N5z//eQDuvfdevvzlL/PUU0+xfPly3J25c+fy5JNPMnr0aH73u99xxx13MHPmTFatWsWmTZt4/vnnAXqtCvXuu+9y0UUXcc8993DiiSfy1ltvMWTIEL797W9jZqxdu5YXXniBc845h5deeqnbe6+77jqmTp3K/fffz89//nMuu+wyVq9eDQRz5ixbtowhQ4aU9ZlFqla+xJG/As6XOKD2boqOHl144ZF+zrXVU7au5NdcezDB5+3bGbSXqdBUw2vXruXhhx9m6tSpTJs2jRdeeIHf/e53ADQ3NzNz5kwA3v/+9/P73/+eq666igcffJDDDjus276LTSe8bNkyLrnkEgA++MEP0tzc3CvJL1u27MCV+llnncXWrVt56623AJg7d64SvGRTDa2tGiqlbprZupLfWeRrTrH2iHpONfzqq6/y1a9+lc997nPdXrdhw4ZuE4YdeeSRrFmzhoceeojvfve73Hvvvdx22239iiUKTVommVVDa6uGyn/zuPbaIP7Ro4MEH/M3kmxdyTcV+ZpTrD2inlMNn3vuudx22228/fbbAGzatIk33nij1/u2bNnC/v37+fjHP84NN9zAM8880217semETz31VNpydcaXXnqJjRs3Mnbs2G7v7fqaJ554gmHDhvX6piCSOQlNJ14xKXTTzNaV/OQbu9fkARqagvZ+6DnV8IgRI1i/fj0nn3wyAIcccghLliyhoaGh2/s2bdrEvHnz2L9/PwDf/OY3u23vOp3wrl27GDJkCI8++igLFy5kwYIFTJw4kcbGRm6//fZui4xAcIP105/+NJMmTaKpqYk77rijX59RpCbceGP3mjzU7EjUtFgwU2V1mDFjhvec3Gv9+vWccMIJ0XfySltQg9+5MbiCn3xj2Tdds6jP51Ok2rS1JV7iqCVmtsrdZxTbnq1yDQQJ/WMb4K/2Bz+V4EVqS1gXySgljjS6WdZIV85slWtEpLbF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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "nQoXkypSfKcN",
        "outputId": "635ba459-8391-4178-8a83-c723d3cbeb41"
      },
      "source": [
        "for i in range(3):\n",
        "    x = df[df['Species'] == species[i]]\n",
        "    plt.scatter(x['Sepal.Width'], x['Petal.Width'], c = colors[i], label=species[i])\n",
        "plt.xlabel(\"Sepal Width\")\n",
        "plt.ylabel(\"Petal Width\")\n",
        "plt.legend()"
      ],
      "execution_count": 58,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7f3a14851290>"
            ]
          },
          "metadata": {},
          "execution_count": 58
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KZsiqaMNok4b"
      },
      "source": [
        "**Training the model**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Hjjm-J8ffn9t"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "# train - 70\n",
        "# test - 30\n",
        "X = df.drop(columns=['Species'])\n",
        "Y = df['Species']\n",
        "x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)"
      ],
      "execution_count": 72,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QxRPUjrEotst"
      },
      "source": [
        "**Using logistic regression**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WUv54gYdfpQJ"
      },
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "model = LogisticRegression()"
      ],
      "execution_count": 70,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Q1ptzwsdfuaB",
        "outputId": "017a05b4-235c-411f-ab27-b48919f058a9"
      },
      "source": [
        "model.fit(x_train, y_train)"
      ],
      "execution_count": 73,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
              "                   intercept_scaling=1, l1_ratio=None, max_iter=100,\n",
              "                   multi_class='auto', n_jobs=None, penalty='l2',\n",
              "                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,\n",
              "                   warm_start=False)"
            ]
          },
          "metadata": {},
          "execution_count": 73
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W93aaOfGfvU4",
        "outputId": "f09f98a3-bace-4bc7-c906-dd6e3a281aaf"
      },
      "source": [
        "print(\"Accuracy: \",model.score(x_test, y_test) )"
      ],
      "execution_count": 76,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy:  0.9904761904761905\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-TV6CB7So157"
      },
      "source": [
        "**using decision tree**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IV7oo8qhfytk"
      },
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "model = DecisionTreeClassifier()"
      ],
      "execution_count": 63,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sWX9yo8of7Np",
        "outputId": "29c26265-bd3b-4953-b3e4-4f9320ca3348"
      },
      "source": [
        "model.fit(x_train, y_train)"
      ],
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',\n",
              "                       max_depth=None, max_features=None, max_leaf_nodes=None,\n",
              "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
              "                       min_samples_leaf=1, min_samples_split=2,\n",
              "                       min_weight_fraction_leaf=0.0, presort='deprecated',\n",
              "                       random_state=None, splitter='best')"
            ]
          },
          "metadata": {},
          "execution_count": 64
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-UpUPSr5gBZD",
        "outputId": "539488cc-a4e6-411d-f0d6-beb6b0c87603"
      },
      "source": [
        "print(\"Accuracy: \",model.score(x_test, y_test) * 100)"
      ],
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy:  100.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iE3sMY1wpaP5"
      },
      "source": [
        "**Using K nearest neighbour**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_JSRVO8opOZT"
      },
      "source": [
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "model = KNeighborsClassifier()"
      ],
      "execution_count": 78,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6y7Ag-BIpOOU",
        "outputId": "7f4e5f72-483f-47bd-cb1f-38596dc3e3db"
      },
      "source": [
        "model.fit(x_train, y_train)"
      ],
      "execution_count": 79,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
              "                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,\n",
              "                     weights='uniform')"
            ]
          },
          "metadata": {},
          "execution_count": 79
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nNlowKRopOCr",
        "outputId": "0d2c599b-5a71-436e-82f3-8363b1f70d8e"
      },
      "source": [
        "print(\"Accuracy: \",model.score(x_test, y_test) * 100)"
      ],
      "execution_count": 80,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy:  99.04761904761905\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rMU4mHINpi1F"
      },
      "source": [
        "**Hence we have sucessfully classified using 3 methods....**"
      ]
    }
  ]
}
