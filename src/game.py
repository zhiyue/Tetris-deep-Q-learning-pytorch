from .tetris2 import Tetris
import time
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import copy

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

  def reset(self):
    self.record = []
    self.score = 0
    self.tetris.reset()
    # self.render()

  def start_auto_run(self, is_init=False):
    if self.tetris.status == 'running':
      self.play_step()
      self.start_auto_run()

  def play_step(self, dir='down', step_count=1, need_update=True, force_update=False):

    bottom = None
    if not force_update:
      # 先执行位移
      bottom = self.tetris.move(dir, step_count)['bottom']

    if ((need_update and bottom == 0) or force_update):
      top_touched, is_round_limited = self.tetris.update()
      if top_touched or is_round_limited:
        max_brick_count, brick_count = self.tetris.max_brick_count, self.tetris.brick_count
        self.game_over()
      else:
        is_valid, brick_count = self.tetris.init_brick()
        if not is_valid:
          self.game_over()
    self.render()

  def get_next_states(self):
    states = {}


  def get_rotate_brick(self):

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
    print(msg)

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