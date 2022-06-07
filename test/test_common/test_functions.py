from nptyping import NDArray
import numpy as np

from dl_scratch.common import functions


def test_identity_function():
    x = 1
    assert np.isclose(functions.identity_function(x), x)

    x = np.array([1, 2, 3])
    assert np.allclose(functions.identity_function(x), x) 

    x = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(functions.identity_function(x), x)


def test_step_function():
    x = np.array([-1, 0, 1, 2])
    assert np.allclose(functions.step_function(x), np.array([0, 0, 1, 1]))

    x = np.array([[-1, 0, 1, 2], [-2.5, -1.3, 0.0, 0.4]])
    assert np.allclose(functions.step_function(x), np.array([[0, 0, 1, 1], [0, 0, 0, 1]]))


def test_sigmoid():
    x = np.array([-2, -1, 0, 1, 2])
    assert np.allclose(functions.sigmoid(x), np.array(([0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708])))

    x = np.array([[-0.5, 0, 0.5], [0.1, 0.2, 0.3]])
    assert np.allclose(functions.sigmoid(x), np.array([[0.37754067, 0.5, 0.62245933], [0.52497919, 0.549834  , 0.57444252]]))


def test_softmax():
    x = np.array([-1, 2, 3])
    assert np.allclose(functions.softmax(x), np.array([0.01321289, 0.26538793, 0.72139918]))
    assert np.isclose(np.sum(functions.softmax(x)), 1.0)

    x = np.array([[-1, 2, 3], [-1, 2, 3]])
    assert np.allclose(functions.softmax(x), np.array([[0.01321289, 0.26538793, 0.72139918], [0.01321289, 0.26538793, 0.72139918]]))
    assert np.allclose(np.sum(functions.softmax(x), axis=-1), np.array([1.0, 1.0]))

    x : NDArray[(2, 2, 3)] = np.array([
        [
            [-1, 2, 3], [-1, 2, 3]
        ], 
        [
            [-1, 2, 3], [-1, 2, 3]
        ]
    ])
    result = functions.softmax(x)
    assert result.shape == (2, 2, 3)
    wanted_value : NDArray[(2, 2, 3)] = np.array([
        [
            [0.01321289, 0.26538793, 0.72139918]
        ],
        [
            [0.01321289, 0.26538793, 0.72139918]
        ]
    ])
    assert np.allclose(functions.softmax(x), wanted_value)


def test_sum_squared_error():
    x, t = np.array([0, 0, 1]), np.array([0, 1, 0])
    assert np.isclose(functions.sum_squared_error(x, t), 1.0)
    
    x, t = np.array([[0, 0, 1], [0, 0, 1]]), np.array([[0, 1, 0], [0, 0, 1]])
    assert np.isclose(functions.sum_squared_error(x, t), 1.0)

    x, t = np.array([[1, 0], [1, 0], [1, 0]]), np.array([[1, 0], [1, 0], [1, 0]])
    assert np.isclose(functions.sum_squared_error(x, t), 0.0)


def test_cross_entropy_error():
    x, t = np.array([0, 0, 1]), np.array([0, 1, 0])
    assert np.isclose(functions.cross_entropy_error(x, t), 16.118095650958317)

    x, t = np.array([[0, 0, 1], [0, 0, 1]]), np.array([[0, 1, 0], [0, 0, 1]])
    assert np.isclose(functions.cross_entropy_error(x, t), 8.059047775479161)

    x, t = np.array([[1, 0], [1, 0], [1, 0]]), np.array([[1, 0], [1, 0], [1, 0]])
    assert np.isclose(functions.cross_entropy_error(x, t), 0, atol=1e-5)

    x, t = np.array([[0, 0, 1], [0, 0, 1]]), np.array([1, 2])
    assert np.isclose(functions.cross_entropy_error(x, t), 8.059047775479161)


