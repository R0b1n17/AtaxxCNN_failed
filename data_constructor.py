import numpy as np
import tensorflow as tf
import random
import hashlib
import os

# -----------------------------
# 加载模型
# -----------------------------
model = tf.keras.models.load_model("ataxx_self_play_mcts_model.h5")
print("模型加载成功。")

# -----------------------------
# Ataxx 棋盘类（基于自我对弈）
# -----------------------------
class AtaxxBitboard:
    SIZE = 7  # 棋盘尺寸

    def __init__(self):
        # 初始化棋盘：0 表示空，1 表示黑棋，-1 表示白棋
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)
        # 黑棋初始位置： (0,0) 和 (6,6)
        # 白棋初始位置： (0,6) 和 (6,0)
        self.black = [(0, 0), (6, 6)]
        self.white = [(0, 6), (6, 0)]
        self.set_piece(0, 0, 1)
        self.set_piece(6, 6, 1)
        self.set_piece(0, 6, -1)
        self.set_piece(6, 0, -1)
        self.current_player = 1  # 黑棋先手

    def set_piece(self, x, y, color):
        self.board[x, y] = color

    def get_legal_moves(self, color):
        """返回当前 color 方所有合法走法，走法表示为 (x0, y0, x1, y1)"""
        moves = []
        pieces = self.black if color == 1 else self.white
        for (x0, y0) in pieces:
            for dx in [-2, -1, 1, 2]:
                for dy in [-2, -1, 1, 2]:
                    x1, y1 = x0 + dx, y0 + dy
                    if 0 <= x1 < self.SIZE and 0 <= y1 < self.SIZE and self.board[x1, y1] == 0:
                        moves.append((x0, y0, x1, y1))
        return moves

    def apply_move(self, move, color):
        """执行走法 move 并更新棋盘和棋子列表；move 为 (x0, y0, x1, y1)"""
        x0, y0, x1, y1 = move
        new_board = self.clone()
        # 判断是否为跳跃（移动距离 2）还是复制（移动距离 1）
        is_jump = (max(abs(x1 - x0), abs(y1 - y0)) == 2)
        if is_jump:
            # 跳跃时移除原始棋子
            new_board.board[x0, y0] = 0
            if color == 1 and (x0, y0) in new_board.black:
                new_board.black.remove((x0, y0))
            elif color == -1 and (x0, y0) in new_board.white:
                new_board.white.remove((x0, y0))
        # 在目标位置放置棋子
        new_board.board[x1, y1] = color
        if color == 1:
            new_board.black.append((x1, y1))
        else:
            new_board.white.append((x1, y1))
        # 翻转目标周围的敌方棋子（8邻域）
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x1 + dx, y1 + dy
                if 0 <= nx < self.SIZE and 0 <= ny < self.SIZE:
                    if color == 1 and new_board.board[nx, ny] == -1:
                        new_board.board[nx, ny] = 1
                        if (nx, ny) in new_board.white:
                            new_board.white.remove((nx, ny))
                        new_board.black.append((nx, ny))
                    elif color == -1 and new_board.board[nx, ny] == 1:
                        new_board.board[nx, ny] = -1
                        if (nx, ny) in new_board.black:
                            new_board.black.remove((nx, ny))
                        new_board.white.append((nx, ny))
        new_board.current_player = -color
        return new_board

    def clone(self):
        new_bb = AtaxxBitboard()
        new_bb.board = self.board.copy()
        new_bb.black = self.black.copy()
        new_bb.white = self.white.copy()
        new_bb.current_player = self.current_player
        return new_bb

    def to_tensor(self):
        """将棋盘转换为 (7,7,1) 的 float32 数组"""
        return self.board.astype(np.float32).reshape(7, 7, 1)

    def get_hash(self):
        """返回当前棋盘状态的唯一哈希（SHA1）"""
        return hashlib.sha1(self.board.tobytes()).hexdigest()

    def display(self):
        for i in range(self.SIZE):
            print(" ".join(['B' if self.board[i, j] == 1 else ('W' if self.board[i, j] == -1 else '.') for j in range(self.SIZE)]))
        print()

    def evaluate(self):
        # 使用简单的局面评估（可以与模型输出做对比）
        black_count = np.count_nonzero(self.board == 1)
        white_count = np.count_nonzero(self.board == -1)
        return black_count - white_count

# -----------------------------
# Minimax 搜索（带 α-β剪枝）——基于搜索算法生成走法
# -----------------------------
def minimax(board, depth, alpha, beta, color):
    if depth == 0:
        return board.evaluate()
    moves = board.get_legal_moves(color)
    if not moves:
        return board.evaluate()
    if color == board.current_player:
        max_eval = -float('inf')
        for move in moves:
            next_board = board.apply_move(move, color)
            eval_score = minimax(next_board, depth - 1, alpha, beta, -color)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            next_board = board.apply_move(move, color)
            eval_score = minimax(next_board, depth - 1, alpha, beta, -color)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def choose_best_move(board, search_depth=3):
    legal_moves = board.get_legal_moves(board.current_player)
    best_move = None
    best_score = -float('inf')
    for move in legal_moves:
        next_board = board.apply_move(move, board.current_player)
        score = minimax(next_board, search_depth - 1, -float('inf'), float('inf'), -board.current_player)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

# -----------------------------
# 利用搜索算法生成自我对弈局面
# -----------------------------
def self_play_generate(search_depth=3, moves_per_game=10):
    board = AtaxxBitboard()
    for _ in range(moves_per_game):
        legal = board.get_legal_moves(board.current_player)
        if not legal:
            break
        # 使用搜索算法选择走法
        move = choose_best_move(board, search_depth)
        if move is None:
            break
        board = board.apply_move(move, board.current_player)
    return board

# -----------------------------
# 将棋盘数据转换为模型输入格式
# -----------------------------
def board_to_input(board):
    # 这里假设黑棋为 1，白棋为 -1，空为 0，转换为 (7,7,1) float32 数组
    return board.to_tensor()

# -----------------------------
# 生成 eval_data.h 文件
# -----------------------------
def generate_eval_data_header(samples, header_filename="eval_data.h"):
    lines = []
    num_samples = len(samples)
    lines.append("#pragma once\n\n")
    lines.append(f"const int NUM_SAMPLES = {num_samples};\n\n")
    lines.append("const float board_states[NUM_SAMPLES][49] = {\n")
    for hash_val, board_flat, score in samples:
        line = "    {" + ", ".join(f"{v:.1f}" for v in board_flat) + "},\n"
        lines.append(line)
    lines.append("};\n\n")
    lines.append("const float model_scores[NUM_SAMPLES] = {\n")
    for hash_val, board_flat, score in samples:
        lines.append(f"    {score:.4f}f,\n")
    lines.append("};\n")
    with open(header_filename, "w") as f:
        f.writelines(lines)
    print(f"{header_filename} 已生成！")

# -----------------------------
# 主函数：生成局面数据并保存 eval_data.h
# -----------------------------
if __name__ == "__main__":
    num_samples = 1000  # 生成 1000 个样本
    samples = []
    for i in range(num_samples):
        board = self_play_generate(search_depth=3, moves_per_game=10)
        tensor = board_to_input(board)  # (7,7,1)
        # 用你的模型评估局面（输入需要扩展 batch 维度）
        input_tensor = np.expand_dims(tensor, axis=0)  # (1,7,7,1)
        score = model.predict(input_tensor)[0][0]
        samples.append((board.get_hash(), tensor.flatten(), score))
    generate_eval_data_header(samples, "eval_data.h")
