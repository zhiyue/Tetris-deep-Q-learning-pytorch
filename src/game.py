from .tetris2 import Tetris
import time
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import copy

gap_action_map = {
  "right": "right",
  "bottom": "down",
  "left": "left",
}

class Game:
  def __init__(self, height=20, width=10, block_size=30):
    self.record = []
    self.score = 0
    self.tetris = Tetris()
    self.height = height
    self.width = width
    self.block_size = block_size
    self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
    self.text_color = (200, 20, 220)

  def start(self):
    self.reset()

    self.tetris.status = 'running'
    self.tetris.init_grids()
    self.tetris.init_brick()
    self.start_auto_run()
    grids = copy.deepcopy(self.tetris.grids)
    return self.get_state_properties(grids, self.tetris.cur_brick_info['pos'])

  def reset(self):
    self.record = []
    self.score = 0
    self.tetris.reset()

    # self.render()

  def start_auto_run(self, is_init=False):
    if self.tetris.status == 'running':
      self.play_step()
      # self.start_auto_run()

  def play_step(self, dir='down', step_count=1, need_update=True, force_update=False):

    bottom = None
    if not force_update:
      # 先执行位移
      gaps, _, _ = self.tetris.move(dir, step_count)
      bottom = gaps['bottom']
    score = 1
    game_over = False
    if ((need_update and bottom == 0) or force_update):
      step_score, top_touched, is_round_limited = self.tetris.update()
      score += step_score
      if top_touched or is_round_limited:
        game_over = True
        self.game_over()
      else:
        is_valid, brick_count = self.tetris.init_brick()
        if not is_valid:
          game_over = True
          self.game_over()
        gaps = self.tetris.get_brick_gaps(self.tetris.cur_brick_info, self.tetris.grids)
        if gaps['bottom'] == 0:
          game_over = True
          self.game_over()
    if self.tetris.status == 'stopped':
      game_over = True
      score -= 2
    self.render()
    return score, game_over

  def step(self, action, render=True):
    (r, action, step_count) = action
    for _ in range(r):
      self.tetris.rotate()
    if action == '':
      score, game_over = self.play_step('', 0, force_update=True)
    else:
      score, game_over = self.play_step(action, step_count)
    self.tetris.tetrominoes += 1
    return score, game_over


  def hash_pos(self, pos):
    hash_str = ''
    for row in pos:
      hash_str += ''.join(str(i) for i in row)
    return hash_str

  def check_cleared_rows(self, grids):
    to_delete = []
    for i, row in enumerate(grids[::-1]):
      if 0 not in row:
        to_delete.append(len(grids) - 1 - i)
    if len(to_delete) > 0:
      grids = self.remove_row(grids, to_delete)
    return len(to_delete), grids

  def remove_row(self, grids, to_delete):
    for index in to_delete:
      grids.pop(index)
      grids.insert(0, [0] * self.width)
    return grids

  def get_state_properties(self, grids, cur_brick_pos):
    for row_index, row in enumerate(grids):
      for col_index, grid in enumerate(row):
        if grid == 0:
          grids[row_index][col_index] = (0,0,0)
    lines_cleared, grids = self.check_cleared_rows(grids)
    holes = self.get_holes(grids)
    bumpiness, height = self.get_bumpiness_and_height(grids)

    return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

  def get_holes(self, board):
    num_holes = 0
    for col in zip(*board):
      row = 0
      while row < self.height and col[row] == 0:
        row += 1
      num_holes += len([x for x in col[row + 1:] if x == 0])
    return num_holes

  def get_bumpiness_and_height(self, board):
    board = np.array(board)
    mask = board != 0
    invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
    heights = self.height - invert_heights
    total_height = np.sum(heights)
    currs = heights[:-1]
    nexts = heights[1:]
    diffs = np.abs(currs - nexts)
    total_bumpiness = np.sum(diffs)
    return total_bumpiness, total_height


  def get_next_states(self):
    states = {}
    positions_seen = set()

    grids = copy.deepcopy(self.tetris.grids)
    # 旋转 3 个方向
    is_valid = True
    cur_brick_info = self.tetris.cur_brick_info
    for r in range(4):
      if not is_valid:
        continue
      cur_brick_pos = [row[:] for row in cur_brick_info['pos']]
      hash_str = self.hash_pos(cur_brick_pos)
      if hash_str in positions_seen:
        continue
      positions_seen.add(hash_str)
      # states[(r, '', 0)] = self.get_state_properties(grids, cur_brick_pos)

      gaps = self.tetris.get_brick_gaps(cur_brick_info, grids)
      gaps.pop('top')
      for direction, max_length in gaps.items():
        action = gap_action_map.get(direction)
        for i in range(1, max_length+1):
          _, is_valid, brick_info = self.tetris.move(action, i, mute=True)
          if not is_valid:
            break
          hash_str = self.hash_pos(brick_info['pos'])
          if hash_str in positions_seen:
            continue
          positions_seen.add(hash_str)
          states[(r, action, i)] = self.get_state_properties(grids, brick_info['pos'])

      is_valid, cur_brick_info = self.tetris.rotate(mute=True)
    if len(states) == 0:
      print(states)
    return states

  def game_over(self):
    op_record, score, brick_count = self.tetris.op_record, self.tetris.score, self.tetris.brick_count
    grids_str, brick_str = self.tetris.get_snapshot()
    self.stop()
    msg = f'''【游戏结束信息】
      当前运行方块数：{brick_count}
      当前得分：{score}
      操作记录: {op_record}
      最后时刻的画布位置信息：（当最后一个砖块的位置合法时，将包含最后一个砖块在内）\n
{grids_str}
      最后时刻的砖块位置信息：
{brick_str}'''
    # print(msg)

  def stop(self):
    self.tetris.status = 'stopped'

  def render(self, video=None):
    grids = copy.deepcopy(self.tetris.grids)
    for row_index, row in enumerate(grids):
      for col_index, grid in enumerate(row):
        if grid == 0:
          grids[row_index][col_index] = (0,0,0)
    cur_brick_info = self.tetris.cur_brick_info
    if cur_brick_info:
      for (x, y) in cur_brick_info['pos']:
        if y > 0:
          grids[y][x] = cur_brick_info['color']

    img = [grid for row in grids for grid in row ]
    img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
    img = img[..., ::-1]
    img = Image.fromarray(img, "RGB")

    img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
    img = np.array(img)
    img[[i * self.block_size for i in range(self.height)], :, :] = 0
    img[:, [i * self.block_size for i in range(self.width)], :] = 0

    img = np.concatenate((img, self.extra_board), axis=1)
    cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
    cv2.putText(img, str(self.score),
                (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

    if video:
        video.write(img)

    cv2.imshow("Deep Q-Learning Tetris", img)
    cv2.waitKey(1)

if __name__ == '__main__':
  Game().start()