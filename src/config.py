
pieces = [
    [
        #  I 型
        [
            [0, 0],
            [0, -1],
            [0, -2],
            [0, 1],
        ],
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [-1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [0, -2],
            [0, 1],
        ],
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [-1, 0],
        ],
    ],
    [
        #  L 型
        [
            [0, 0],
            [0, -1],
            [0, -2],
            [1, 0],
        ],
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
        ],
        [
            [0, 0],
            [-1, 0],
            [0, 1],
            [0, 2],
        ],
        [
            [0, 0],
            [0, -1],
            [-1, 0],
            [-2, 0],
        ],
    ],
    [
        #  J 型
        [
            [0, 0],
            [0, -1],
            [0, -2],
            [-1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [1, 0],
            [2, 0],
        ],
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [0, 2],
        ],
        [
            [0, 0],
            [-1, 0],
            [-2, 0],
            [0, 1],
        ],
    ],
    [
        #  T 型
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [0, 1],
            [-1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [1, 0],
            [-1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [1, 0],
            [0, 1],
        ],
    ],
    [
        #  O 型
        [
            [0, 0],
            [0, -1],
            [1, -1],
            [1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [1, -1],
            [1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [1, -1],
            [1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [1, -1],
            [1, 0],
        ],
    ],
    [
        #  S 型
        [
            [0, 0],
            [0, -1],
            [1, -1],
            [-1, 0],
        ],
        [
            [0, 0],
            [-1, 0],
            [-1, -1],
            [0, 1],
        ],
        [
            [0, 0],
            [0, -1],
            [1, -1],
            [-1, 0],
        ],
        [
            [0, 0],
            [-1, 0],
            [-1, -1],
            [0, 1],
        ],
    ],
    [
        #  Z 型
        [
            [0, 0],
            [0, -1],
            [1, 0],
            [-1, -1],
        ],
        [
            [0, 0],
            [0, -1],
            [-1, 1],
            [-1, 0],
        ],
        [
            [0, 0],
            [0, -1],
            [1, 0],
            [-1, -1],
        ],
        [
            [0, 0],
            [0, -1],
            [-1, 1],
            [-1, 0],
        ],
    ],
]

grid_config = {
    "width": 200,
    "height": 400,
    "row": 20,
    "col": 10,
}

piece_colors = [
    (0, 0, 0),
    (255, 255, 0),
    (147, 88, 254),
    (54, 175, 144),
    (255, 0, 0),
    (102, 217, 238),
    (254, 151, 32),
    (0, 0, 255)
]