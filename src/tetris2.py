import numpy as np
import re
from .config import pieces, grid_config, piece_colors
class Tetris:
    default_brick_center_pos = [4, 0]
    random_config = {
        'a': 27073, # 乘子
        'M': 32749, # 模数
        'C': 17713, # 增量
        'v': 12358, # 随机数种子
    }
    action_map = {
      "down": 'D',
      "left": "L",
      "right": "R",
      "rotate": "C",
      "new": "N",
    }

    def __init__(self, height=20, width=10, block_size=20):
        self.shape_index = 0
        self.state_index = 0
        self.grids = []
        self.brick_count = 0
        self.cur_random_num = self.random_config['v']
        self.max_brick_count = 10000
        self.cur_brick_center_pos = None
        self.cur_brick_raw_info = {"pos": None, "color": ""}
        self.cur_brick_info = {}
        self.next_brick_raw_info = {}
        self.score = 0
        self.status = 'stopped'
        self.op_record = []



        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                    dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        self.text_color = (200, 20, 220)
        self.reset()


    def reset(self):
        self.shape_index = 0
        self.state_index = 0
        self.grids = []
        self.brick_count = 0
        self.cur_random_num = self.random_config['v']

        self.next_brick_raw_info = {}
        self.score = 0
        self.status = 'starting'
        self.op_record = []
        self.clear_grids()

    def clear_grids(self):
      self.clear_brick()
      self.grids = [[''] * grid_config['col'] for i in range(grid_config['row'])]

    def clear_brick(self):
        self.cur_brick_center_pos = None
        self.cur_brick_raw_info = {"pos": None, "color": ""}
        self.cur_brick_info = {}

    def get_init_grids(self):
      return [[''] * grid_config['col'] for i in range(grid_config['row'])]

    def init_grids(self):
      self.grids = self.get_init_grids()

    def get_random_num(self, cur_random_num):
      return (cur_random_num * self.random_config['a'] + self.random_config['C']) % self.random_config['M']

    def get_shape_info(self, random_num, brick_count):
      weight_index = random_num % 29;
      state_index = brick_count % len(pieces[0])
      color_index = brick_count % len(piece_colors)
      if  0 <= weight_index <= 1:
        shape_index = 0
      elif 2 <= weight_index <= 4:
        shape_index = 1
      elif 5 <= weight_index <= 7:
        shape_index = 2
      elif 8 <= weight_index <= 11:
        shape_index = 3
      elif 12 <= weight_index <= 16:
        shape_index = 4
      elif 17 <= weight_index <= 22:
        shape_index = 5
      elif 23 <= weight_index <= 29:
        shape_index = 6

      return (shape_index, state_index, color_index)

    def get_raw_brick(self, random_num, brick_count, mute):
      shape_index, state_index, color_index = self.get_shape_info(random_num, brick_count)
      if not mute:
        self.shape_index = shape_index
        self.state_index = state_index
      return {"pos": pieces[shape_index][state_index], "color": piece_colors[color_index]}

    def is_brick_pos_valid(self, brick_info):
      row = grid_config['row']
      col = grid_config['col']
      x_range = [0, col - 1]
      y_range = [0, row - 1]
      valid_count = 0
      for x, y in brick_info['pos']:
        is_horizontal_valid = x_range[0] <= x <= x_range[1]
        is_vertical_valid = y <= y_range[1]
        is_cur_grid_valid = y < 0 or (self.grids[y] and not self.grids[y][x])
        valid_count += 1 if is_horizontal_valid and is_vertical_valid and is_cur_grid_valid else 0
      return valid_count == 4

    def get_brick_position(self, brick_raw_info, brick_center_pos, force_update=False):
      x, y = brick_center_pos
      calced = [[x1+x, y1+y] for x1, y1 in brick_raw_info['pos']]
      brick_info = {
        "pos": calced,
        "color": piece_colors[self.brick_count % len(piece_colors)]
      }
      if self.is_brick_pos_valid(brick_info):
        return True, brick_info
      return False, brick_info if force_update else self.cur_brick_info

    def get_brick_info(self, random_num, brick_count, brick_center_pos, mute=False):
      brick_raw_info = self.get_raw_brick(random_num, brick_count, mute)
      is_valid, brick_info = self.get_brick_position(brick_raw_info, brick_center_pos)
      return is_valid, brick_raw_info, brick_info

    def get_op_info(self, op_cmd):
      op_cmd = op_cmd.strip()
      pattern = re.compile(r'^([LRDCN])(\d*)$')
      op_type = ''
      count = ''
      if pattern.search(op_cmd):
        op_type, count = pattern.search(op_cmd).groups()
      return op_type, int(count)

    def track_op(self, op_type, step_count=1):
      if self.status != 'running':
        return
      op_type = self.action_map[op_type]
      pre_op = self.opRecord[-1] if self.opRecord else None
      pre_op_type, pre_count = self.get_op_info(pre_op) if pre_op else (None, None)
      if pre_op_type == op_type and pre_op_type != 'N':
        self.op_record[-1] = f'{ pre_op_type }{ pre_count+1 }'
      elif op_type == 'D' and step_count > 1:
        self.op_record.append(f'D{step_count}')
      else:
        self.op_record.append(f'{op_type}{"1" if op_type != "N" else ""}')



    def init_brick(self):
      self.cur_random_num = self.get_random_num(self.cur_random_num)
      cur_brick_center_pos = self.default_brick_center_pos.copy()
      is_valid, brick_raw_info, brick_info = self.get_brick_info(self.cur_random_num, self.brick_count, cur_brick_center_pos, False)
      _, next_brick_raw_info, _ = self.get_brick_info(
        self.get_random_num(self.cur_random_num), self.brick_count + 1, cur_brick_center_pos, True)
      self.cur_brick_center_pos = cur_brick_center_pos
      self.cur_brick_raw_info = brick_raw_info
      self.cur_brick_info = brick_info
      self.brick_count += 1
      self.track_op('new')
      if is_valid:
        self.next_brick_raw_info = next_brick_raw_info
      return is_valid, self.brick_count
