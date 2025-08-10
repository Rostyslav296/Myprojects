import chess
import chess.pgn

def print_board(board):
    """Prints the chess board in ASCII format."""
    pieces = {
        chess.PAWN: 'P', chess.KNIGHT: 'N', chess.BISHOP: 'B',
        chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K'
    }
    empty = '.'
    ranks = str(board).split('\n')
    print('  +---+---+---+---+---+---+---+---+')
    for i, rank in enumerate(ranks):
        print(f'{8 - i} |', end=' ')
        for square in rank.split():
            if square.isdigit():
                for _ in range(int(square)):
                    print(empty, end=' | ')
            else:
                piece = square.upper() if square.isupper() else square.lower()
                print(piece if square.isupper() else piece, end=' | ')
        print('\n  +---+---+---+---+---+---+---+---+')
    print('    a   b   c   d   e   f   g   h')
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

def get_move(board):
    """Gets a valid move from the user in UCI format (e.g., e2e4)."""
    while True:
        move_str = input("Enter your move (UCI format, e.g., e2e4) or 'q' to quit: ").strip()
        if move_str.lower() == 'q':
            return None
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move. Try again.")
        except ValueError:
            print("Invalid move format. Use UCI like e2e4.")

def main():
    board = chess.Board()
    print("Welcome to Interactive Chess Game!")
    print("Players alternate turns. White starts.")
    print("Enter moves in UCI format (source square to target square, e.g., e2e4).")
    print("For promotion, add the piece letter, e.g., e7e8q for queen.")
    
    while True:
        print_board(board)
        
        if board.is_game_over():
            if board.is_checkmate():
                winner = "Black" if board.turn == chess.WHITE else "White"
                print(f"Checkmate! {winner} wins.")
            elif board.is_stalemate():
                print("Stalemate! It's a draw.")
            elif board.is_insufficient_material():
                print("Insufficient material! It's a draw.")
            else:
                print("Game over! It's a draw.")
            break
        
        move = get_move(board)
        if move is None:
            print("Game quit.")
            break
        board.push(move)

if __name__ == "__main__":
    main()