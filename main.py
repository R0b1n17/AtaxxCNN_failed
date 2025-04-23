import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
import pickle
import gc
import time

# ---------------------------
# GPU 显存按需分配（如果有 GPU）
# ---------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print("⚠️ Could not set GPU memory growth:", e)
else:
    print("⚠️ No GPU found, running on CPU.")


# ---------------------------
# Ataxx 位板棋盘表示（混合方式）
# ---------------------------
class AtaxxBitboard:
    SIZE = 7  # 棋盘尺寸

    def __init__(self):
        # 用 7x7 数组保存棋盘状态，初始全部为空（0 表示空，1 表示黑棋，-1 表示白棋）
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)
        # 用列表记录棋子位置（方便生成合法走法）
        self.black = [(0, 0), (6, 6)]  # 黑棋初始位置
        self.white = [(0, 6), (6, 0)]  # 白棋初始位置
        # 更新棋盘状态：设置初始棋子
        self.set_piece(0, 0, 1)
        self.set_piece(6, 6, 1)
        self.set_piece(0, 6, -1)
        self.set_piece(6, 0, -1)
        self.current_player = 1  # 黑棋先手

    def set_piece(self, x, y, color):
        """ 在 (x,y) 位置放置棋子；color 为 1 表示黑棋，-1 表示白棋，0 表示清空 """
        self.board[x, y] = color

    def get_legal_moves(self, color):
        """ 返回当前 color 方所有合法走法，走法表示为 (x0, y0, x1, y1) """
        legal_moves = []
        pieces = self.black if color == 1 else self.white
        for (x0, y0) in pieces:
            for dx in [-2, -1, 1, 2]:
                for dy in [-2, -1, 1, 2]:
                    x1, y1 = x0 + dx, y0 + dy
                    if 0 <= x1 < self.SIZE and 0 <= y1 < self.SIZE and self.board[x1, y1] == 0:
                        legal_moves.append((x0, y0, x1, y1))
        return legal_moves

    def apply_move(self, move, color):
        """ 执行走法 move 并更新棋盘和棋子列表 """
        x0, y0, x1, y1 = move
        self.board[x1, y1] = color  # 将目标位置设置为当前 color
        self.board[x0, y0] = 0  # 清空原位置
        if color == 1:
            if (x0, y0) in self.black:
                self.black.remove((x0, y0))
            self.black.append((x1, y1))
        else:
            if (x0, y0) in self.white:
                self.white.remove((x0, y0))
            self.white.append((x1, y1))
        self.current_player = -color

    def evaluate(self):
        """
        改进的评估函数，综合考虑棋子数差、移动性、角落控制、边缘控制和潜在翻转
        """
        # 棋子数差
        piece_diff = len(self.black) - len(self.white)
        # 移动性
        legal_moves_black = len(self.get_legal_moves(1))
        legal_moves_white = len(self.get_legal_moves(-1))
        mobility = legal_moves_black - legal_moves_white
        # 角落控制
        corners = [(0, 0), (0, self.SIZE - 1), (self.SIZE - 1, 0), (self.SIZE - 1, self.SIZE - 1)]
        corner_black = sum(1 for pos in corners if self.board[pos[0], pos[1]] == 1)
        corner_white = sum(1 for pos in corners if self.board[pos[0], pos[1]] == -1)
        corner_control = corner_black - corner_white
        # 边缘控制（不包含角落）
        edge_positions = []
        for i in range(1, self.SIZE - 1):
            edge_positions.append((0, i))
            edge_positions.append((self.SIZE - 1, i))
            edge_positions.append((i, 0))
            edge_positions.append((i, self.SIZE - 1))
        edge_black = sum(1 for pos in edge_positions if self.board[pos[0], pos[1]] == 1)
        edge_white = sum(1 for pos in edge_positions if self.board[pos[0], pos[1]] == -1)
        edge_control = edge_black - edge_white

        # 潜在翻转：计算每个己方棋子周围可能翻转对方棋子的数量
        def potential_flips(color):
            count = 0
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    if self.board[i, j] == color:
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                ni, nj = i + dx, j + dy
                                if 0 <= ni < self.SIZE and 0 <= nj < self.SIZE and self.board[ni, nj] == -color:
                                    count += 1
            return count

        potential_flip = potential_flips(1) - potential_flips(-1)

        # 权重设置（可调）
        w_piece = 10
        w_mobility = 5
        w_corner = 25
        w_edge = 5
        w_flip = 3

        return w_piece * piece_diff + w_mobility * mobility + w_corner * corner_control + w_edge * edge_control + w_flip * potential_flip

    def clone(self):
        """ 克隆当前棋盘状态（深拷贝），用于模拟走法而不影响原局面 """
        new_board = AtaxxBitboard()
        new_board.board = self.board.copy()
        new_board.black = self.black.copy()
        new_board.white = self.white.copy()
        new_board.current_player = self.current_player
        return new_board

    def display(self):
        """ 打印棋盘状态，B 表示黑棋，W 表示白棋，. 表示空位 """
        for i in range(self.SIZE):
            row = []
            for j in range(self.SIZE):
                if self.board[i, j] == 1:
                    row.append('B')
                elif self.board[i, j] == -1:
                    row.append('W')
                else:
                    row.append('.')
            print(" ".join(row))
        print()


# ---------------------------
# MCTS 搜索（批量评估所有合法走法）
# ---------------------------
class MCTS:
    def __init__(self, board, color, model):
        self.board = board  # 当前棋盘（AtaxxBitboard 对象）
        self.color = color  # 当前搜索的棋方（1 或 -1）
        self.model = model  # 用于评估局面的神经网络模型

    def search_best_move(self):
        legal_moves = self.board.get_legal_moves(self.color)
        if not legal_moves:
            return None

        board_states = []
        moves = []
        # 对每个合法走法，克隆棋盘后执行该走法，收集局面
        for move in legal_moves:
            clone_board = self.board.clone()
            clone_board.apply_move(move, self.color)
            board_states.append(clone_board.board.copy())
            moves.append(move)

        # 合并为批次：扩展维度为 (7,7,1)
        batch_input = np.stack(board_states, axis=0).reshape(len(board_states), 7, 7, 1).astype(np.float32)
        start_time = time.time()
        predictions = self.model.predict(batch_input, batch_size=16, verbose=0)
        end_time = time.time()
        print(f"批量预测耗时: {end_time - start_time:.4f} 秒")
        best_value = -float('inf')
        best_move = None
        for i, pred in enumerate(predictions):
            if pred[0] > best_value:
                best_value = pred[0]
                best_move = moves[i]
        return best_move


# ---------------------------
# 神经网络模型构建
# ---------------------------
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='tanh')  # 输出评估值，范围 -1 到 1
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ---------------------------
# 自定义回调：输出训练进度
# ---------------------------
class PrintProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}: loss = {logs.get('loss'):.4f}, "
              f"mae = {logs.get('mae'):.4f}, "
              f"val_loss = {logs.get('val_loss'):.4f}, "
              f"val_mae = {logs.get('val_mae'):.4f}")


# ---------------------------
# 自我对弈数据生成函数（新规则：如果出现重复局面则判负）
# ---------------------------
def self_play(model, num_games=100):
    game_data = []
    for game_idx in range(num_games):
        board = AtaxxBitboard()
        seen_states = set()  # 存储已经出现过的局面（字符串形式）
        mcts = MCTS(board, color=1, model=model)
        game_history = []
        step = 0
        while True:
            # 获取当前局面状态字符串
            state_str = board.board.tostring()  # 或者 board.board.tobytes()
            if state_str in seen_states:
                # 如果局面重复，则判负：记录一个极端评估值，例如 -1000，并结束游戏
                print(f"重复局面出现，游戏 {game_idx + 1} 在步 {step} 结束")
                game_history.append((board.board.copy(), None, -1000))
                break
            seen_states.add(state_str)
            move = mcts.search_best_move()
            if move is None:
                break
            step += 1
            if step % 10 == 0:
                print(f"游戏 {game_idx + 1}, 步数 {step}, 当前评估值 {board.evaluate()}")
            game_history.append((board.board.copy(), move, board.evaluate()))
            board.apply_move(move, board.current_player)
        print(f"游戏 {game_idx + 1}完成，共 {step} 步")
        game_data.append(game_history)
        gc.collect()
    return game_data


# ---------------------------
# 主函数
# ---------------------------
def main():
    model = build_model((7, 7, 1))  # 构建模型
    model.summary()

    total_games = 2000  # 总数据量 2000 局
    batch_size = 100  # 每批生成 100 局
    num_batches = total_games // batch_size
    all_game_data = []

    for batch_idx in range(1, num_batches + 1):
        filename = f"game_data_batch_{batch_idx}.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                batch_data = pickle.load(f)
            all_game_data.extend(batch_data)
            print(f"加载已存在的第 {batch_idx} 批数据 {filename}")
        else:
            print(f"生成第 {batch_idx} 批 {batch_size} 局自我对弈数据...")
            batch_data = self_play(model, num_games=batch_size)
            all_game_data.extend(batch_data)
            with open(filename, "wb") as f:
                pickle.dump(batch_data, f)
            print(f"第 {batch_idx} 批数据已保存为 {filename}")
            gc.collect()

    # 生成训练样本：使用棋盘状态作为输入，局面评估值作为目标
    X_train = []
    y_train = []
    for game in all_game_data:
        for board_state, move, reward in game:
            X_train.append(board_state)
            y_train.append(reward)
    X_train = np.array(X_train).reshape(-1, 7, 7, 1).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint('ataxx_self_play_mcts_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    print_progress = PrintProgress()

    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=10, batch_size=32, verbose=2,
                        callbacks=[early_stop, checkpoint, print_progress])

    model.save('ataxx_self_play_mcts_model.h5')
    print("模型训练完成并保存为 ataxx_self_play_mcts_model.h5")

    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
