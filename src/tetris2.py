import copy
import re

import numpy as np

from .config import grid_config, piece_colors, pieces


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
    occupied_count_score_map = {
      1: 1,
      2: 3,
      3: 6,
      4: 10,
    }

    def __init__(self):
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
      self.grids = [[0] * grid_config['col'] for i in range(grid_config['row'])]

    def clear_brick(self):
        self.cur_brick_center_pos = None
        self.cur_brick_raw_info = {"pos": None, "color": ""}
        self.cur_brick_info = {}

    def get_init_grids(self):
      return [[0] * grid_config['col'] for i in range(grid_config['row'])]

    def init_grids(self):
      self.grids = self.get_init_grids()

    def get_random_num(self, cur_random_num):
      return (cur_random_num * self.random_config['a'] + self.random_config['C']) % self.random_config['M']

    def get_shape_info(self, random_num, brick_count):
      weight_index = random_num % 29
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
      print(brick_info)
      row = grid_config['row']
      col = grid_config['col']
      x_range = [0, col - 1]
      y_range = [0, row - 1]
      valid_count = 0
      for x, y in brick_info['pos']:
        is_horizontal_valid = x_range[0] <= x <= x_range[1]
        is_vertical_valid = y <= y_range[1]
        is_cur_grid_valid = y < 0 or (len(self.grids) >= y+1 and self.grids[y] and len(self.grids[y]) >= x+1 and not self.grids[y][x])
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
      count = 0
      if pattern.search(op_cmd):
        op_type, count = pattern.search(op_cmd).groups()
        if count == '':
          count = 1
      return op_type, int(count)

    def track_op(self, op_type, step_count=1):
      """记录方块相关的历史操作
      :param op_type: 动作类型
      :param step_count: 步数
      """
      if self.status != 'running':
        return
      op_type = self.action_map[op_type]
      pre_op = self.op_record[-1] if self.op_record else None
      pre_op_type, pre_count = self.get_op_info(pre_op) if pre_op else (None, None)
      if pre_op_type == op_type and pre_op_type != 'N':
        self.op_record[-1] = f'{ pre_op_type }{ pre_count+1 }'
      elif op_type == 'D' and step_count > 1:
        self.op_record.append(f'D{step_count}')
      else:
        self.op_record.append(f'{op_type}{"1" if op_type != "N" else ""}')



    def init_brick(self):
      """初始化方块"""
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

    def update(self):
      """
      当一个方块落定，更新格子（是否有消除行）、分数，并返回是否堆叠触顶或者超出允许的最大方块数量
      """
      pos, color = self.cur_brick_info['pos'], self.cur_brick_info['color']
      for x, y in pos:
        if self.grids[y]:
          self.grids[y][x] = color
      full_row_indexes = []
      occupied_grid_count = 0
      min_occupied_row_index = grid_config['row'] - 1
      for row_index, row in enumerate(self.grids):
        occupied_grird_count_per_row = 0

        # 每行已占用的格子数
        for grid in row:
          if grid != 0:
            occupied_grird_count_per_row += 1

        # 当前行有被占用的格子， 被占用行计数加 1
        if occupied_grird_count_per_row > 0:
          min_occupied_row_index = row_index if row_index < min_occupied_row_index else min_occupied_row_index

        # 当前行所有格子都被占用，满行计数加1
        if occupied_grird_count_per_row == len(row):
          full_row_indexes.append(row_index)

        occupied_grid_count += occupied_grird_count_per_row
      ret = {
        "top_touched": min_occupied_row_index == 0,
        "is_round_limited": self.brick_count >= self.max_brick_count
      }

      # 触顶或者超过游戏的最大方块数量时，不计分数
      if ret["top_touched"] or ret["is_round_limited"]:
        return ret["top_touched"], ret["is_round_limited"]


      # 分数计算规则（富贵险中求）：界面上堆砌的格子数乘以当前消除行数的得分系数
      # 当前消除行的得分系数：消除的行数越多，系数随之增加
      # 如：当前I型方块消除的行数为 18,17,15 共 3 行，则得分为 occupiedGridCount * 6
      score = occupied_grid_count * self.occupied_count_score_map.get(len(full_row_indexes), 0)

      for index in full_row_indexes:
        # 把对应行清除，并且在顶部加入新的一行
        self.grids.pop(index)
        self.grids.insert(0, [0] * grid_config['col'])

      self.score += score
      return ret['top_touched'], ret['is_round_limited']

    def get_brick_gaps(self, brick_info, grids):
      """
       获取当前方块在画布上各方向的格子间隙（也即各方向还能移动多少格)
      :param brick_info: 方块信息
      :param grids: 画布网格信息
      :return:
      """
      ret = {
        "top": grid_config['row'],
        "right": grid_config['col'],
        "bottom": grid_config['row'],
        "left": grid_config['col'],
      }
      for x, y in brick_info['pos']:
        cur_gaps = {
          "top": 0,
          "right": 0,
          "bottom": 0,
          "left": 0,
        }

        # 左侧计算
        for i in range(x - 1, -1, -1):
          if grids[y][i] != 0:
            break
          cur_gaps['left'] += 1

        # 右侧计算
        for i in range(x + 1, grid_config['col']):
          if grids[y][i] != 0:
            break
          cur_gaps['right'] += 1

        # 上侧计算
        for i in range(y - 1, -1, -1):
          if i < 0:
            continue
          if grids[i][x] != 0:
            break
          cur_gaps['top'] += 1

        # 下侧计算
        for i in range(y + 1, grid_config['row']):
          if i < 0:
            continue
          if grids[i][x] != 0:
            break
          cur_gaps['bottom'] += 1

        for k, v in cur_gaps.items():
          ret[k] = min(ret[k], v)

      return ret

    def move(self, dir, step_count=1):
      """
      方块移动, 并返回移动后各方向的空格间隙
      :param dir: 方向
      :param step_count: 移动步数
      :return:
      """

      center_pos = copy.deepcopy(self.cur_brick_center_pos)
      if dir == 'left':
        center_pos[0] -= step_count
      elif dir == 'right':
        center_pos[0] += step_count
      elif dir == 'down':
        center_pos[1] += step_count

      is_valid, brick_info = self.get_brick_position(self.cur_brick_raw_info, center_pos)
      gaps = self.get_brick_gaps(self.cur_brick_info, self.grids)
      if is_valid:
        self.cur_brick_info = brick_info
        self.cur_brick_center_pos = center_pos
        self.track_op(dir, step_count)
      return gaps

    def rotate(self, mute=False):
      """
       旋转方块（实际为按照 stateIndex 渲染对应的方块形态）
      :return:
      """
      state_index = self.state_index
      if self.state_index >= 3:
        self.state_index = 0
      else:
        state_index += 1

      cur_brick_raw_info = {
        "pos": pieces[self.shape_index][state_index],
        "color": self.cur_brick_raw_info['color'],
      }

      is_valid, brick_info = self.get_brick_position(cur_brick_raw_info, self.cur_brick_center_pos)
      if is_valid:
        self.state_index = state_index
        self.cur_brick_info = brick_info
        self.cur_brick_raw_info = cur_brick_raw_info
        self.track_op('rotate')

    def drop(self):
      """
      方块下落
      :return:
      """
      bottom = self.get_brick_gaps(self.cur_brick_info, self.grids)['bottom']
      self.move('down', bottom)

    def get_snapshot(self):
      """
      获取当前游戏状态的快照（每个方块的信息）
      :return:
      """
      grids, cur_brick_info = self.grids, self.cur_brick_info
      grids_str = '     0  1  2  3  4  5  6  7  8  9 \n     ----------------------------\n';
      brick_str = '     0  1  2  3  4  5  6  7  8  9 \n     ----------------------------\n';
      for row_index, row in enumerate(grids):
        head = f'{row_index}'
        head = head.ljust(2, '0')
        grids_str += f'{head} |'
        brick_str += f'{head} |'
        for col_index, grid in enumerate(row):
           grids_str += ' # ' if grid else ' . ';
           is_brick_pos = cur_brick_info.get('pos') and [col_index, row_index] in cur_brick_info['pos']
           brick_str += ' # ' if is_brick_pos else ' . ';
        grids_str += '\n'
        brick_str += '\n'
      return grids_str, brick_str