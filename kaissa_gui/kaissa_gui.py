"""
GUI for Southern Kaissa with:
- Strong predictor (press 'S' for green best-move hint)
- Blunder overlays (press 'E' after selecting a piece):
    * RED border on a destination if the moved piece will die immediately
      + red ARROWS from enemy piece(s) to that destination
    * ORANGE border if some other piece will die immediately
      + orange ARROWS from enemy piece(s) to those victim squares
- Capturable overlay (press 'C'):
    * Shows every square that is capturable by either side
    * RED ARROWS from Red attackers to Yellow victims
    * YELLOW ARROWS from Yellow attackers to Red victims
"""

import math
import pygame
import sys
import os
from pygame.locals import VIDEORESIZE

from kaissa_engine.kaissa_engine import get_protected_squares, index_to_rc
from kaissa_engine.kaissa_engine import (
    KaissaGameEngine, BOARD_ROWS, BOARD_COLS,
    UBAR, UBARA, TARNSMAN, BUILDER, INITIATE,
    SCRIBE, ASSASSIN, RIDER, SPEARMAN, HOMESTONE
)

# Strong predictor
from kaissa_ai.stronger_predictor import predict_best_move_strong

# Blunder checker (borders + arrow sources)
from kaissa_ai.blunder_evaluator import classify_destinations_for_selected_piece

# Minimum window dimensions
MIN_WINDOW_WIDTH = 600
MIN_WINDOW_HEIGHT = 400

# Panels ratios
SIDE_PANEL_RATIO = 0.10
TOP_HUD_RATIO    = 0.10

# Colours
COLOR_YELLOW_PIECE  = (255, 220, 100)
COLOR_RED_PIECE     = (220, 60, 60)
CIRCLE_BORDER_COLOR = (0, 0, 0)

COLOR_LIGHT         = (240, 217, 181)
COLOR_DARK          = (181, 136,  99)
COLOR_HIGHLIGHT     = (186, 202, 68)
COLOR_SELECTED      = (246, 246, 105)
COLOR_BACKGROUND    = (40, 40, 40)
COLOR_TEXT          = (0, 0, 0)
COLOR_HUD_BG        = (100, 100, 100)
COLOR_HUD_TEXT      = (255, 255, 255)

COLOR_SHOW_RED      = (255, 80, 80)      # move bubbles & red-side arrows in C-view
COLOR_SHOW_YELLOW   = (255, 255, 150)    # move bubbles & yellow-side arrows in C-view
COLOR_ALL_CAPTURES  = (255, 0, 0)        # red overlay on capturable squares

COLOR_BLUNDER_RED    = (255, 0, 0)       # moved piece dies (E-view)
COLOR_BLUNDER_ORANGE = (255, 165, 0)     # other piece dies (E-view)

# Fallback symbols
PIECE_SYMBOLS = {
    (UBAR, True): "U",       (UBAR, False): "U",
    (UBARA, True): "V",      (UBARA, False): "V",
    (TARNSMAN, True): "T",   (TARNSMAN, False): "T",
    (BUILDER, True): "B",    (BUILDER, False): "B",
    (INITIATE, True): "I",   (INITIATE, False): "I",
    (SCRIBE, True): "S",     (SCRIBE, False): "S",
    (ASSASSIN, True): "A",   (ASSASSIN, False): "A",
    (RIDER, True): "R",      (RIDER, False): "R",
    (SPEARMAN, True): "P",   (SPEARMAN, False): "P",
    (HOMESTONE, True): "H",  (HOMESTONE, False): "H",
}

PIECE_NAME_MAP = {
    UBAR:      "ubar",
    UBARA:     "ubara",
    TARNSMAN:  "tarnsman",
    BUILDER:   "builder",
    INITIATE:  "initiate",
    SCRIBE:    "scribe",
    ASSASSIN:  "assassin",
    RIDER:     "rider",
    SPEARMAN:  "spearman",
    HOMESTONE: "homestone"
}

class KaissaGUI:
    def __init__(self):
        pygame.init()
        self.window_width  = 1000
        self.window_height = 600

        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Southern Kaissa (Enhanced GUI)")

        self.game_engine = KaissaGameEngine()

        # Suggestions
        self.best_move = None
        self.best_overall = None

        # Selection
        self.selected_square = None
        self.legal_moves_for_selected = []

        # Captures panel
        self.red_captures = []
        self.yellow_captures = []
        self.half_move_count = 0

        self.running = True
        self.clock = pygame.time.Clock()

        self.board_flipped = False
        self.show_all_moves_red = False
        self.show_all_moves_yellow = False
        self.show_capturable_pieces = False

        # Blunder overlay state (E)
        self.show_blunders_for_selected = False
        self.blunder_red_dests = set()
        self.blunder_orange_dests = set()
        self.blunder_red_arrows = set()      # {((fr,fc),(tr,tc))}
        self.blunder_orange_arrows = set()   # {((fr,fc),(tr,tc))}

        # Fonts
        self.font_small  = pygame.font.SysFont(None, 20)
        self.font_medium = pygame.font.SysFont(None, 32)
        self.font_large  = pygame.font.SysFont(None, 60)

        self.piece_images = self._load_piece_images()
        self._calc_layout()

    # --- Init helpers ---
    def _load_piece_images(self):
        images_dict = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_path = os.path.join(script_dir, "Images")
        for piece_type, piece_name in PIECE_NAME_MAP.items():
            filename = f"{piece_name}.png"
            full_path = os.path.join(images_path, filename)
            if os.path.exists(full_path):
                img = pygame.image.load(full_path).convert_alpha()
                images_dict[piece_type] = img
            else:
                images_dict[piece_type] = None
        return images_dict

    def _calc_layout(self):
        if self.window_width < MIN_WINDOW_WIDTH:
            self.window_width = MIN_WINDOW_WIDTH
        if self.window_height < MIN_WINDOW_HEIGHT:
            self.window_height = MIN_WINDOW_HEIGHT

        self.left_side_width  = int(self.window_width * SIDE_PANEL_RATIO)
        self.right_side_width = int(self.window_width * SIDE_PANEL_RATIO)
        self.top_hud_height   = int(self.window_height * TOP_HUD_RATIO)

        board_area_width  = self.window_width - self.left_side_width - self.right_side_width
        board_area_height = self.window_height - self.top_hud_height

        square_width  = board_area_width / BOARD_COLS
        square_height = board_area_height / BOARD_ROWS
        self.square_size = int(min(square_width, square_height))

        self.board_draw_width  = self.square_size * BOARD_COLS
        self.board_draw_height = self.square_size * BOARD_ROWS

        self.board_left = self.left_side_width + (board_area_width - self.board_draw_width)//2
        self.board_top  = self.top_hud_height + (board_area_height - self.board_draw_height)//2

    # --- Main loop ---
    def run(self):
        while self.running:
            self.clock.tick(30)
            self.handle_events()
            self.draw()
            pygame.display.flip()
            if self.game_engine.is_game_over():
                pass
        pygame.quit()
        sys.exit()

    # --- Events ---
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == VIDEORESIZE:
                self.window_width, self.window_height = event.w, event.h
                self._calc_layout()
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                elif event.key == pygame.K_n:
                    self._reset_game()
                elif event.key == pygame.K_f:
                    self.board_flipped = not self.board_flipped
                    self._clear_selection_and_overlays()
                elif event.key == pygame.K_r:
                    self.show_all_moves_red = not self.show_all_moves_red
                elif event.key == pygame.K_y:
                    self.show_all_moves_yellow = not self.show_all_moves_yellow
                elif event.key == pygame.K_c:
                    self.show_capturable_pieces = not self.show_capturable_pieces
                elif event.key == pygame.K_s:
                    self.best_overall = predict_best_move_strong(
                        self.game_engine, max_depth=4, time_limit=5.0
                    )
                elif event.key == pygame.K_e:
                    # Toggle blunder view for the selected piece
                    if self.selected_square is not None and self.legal_moves_for_selected:
                        self.show_blunders_for_selected = not self.show_blunders_for_selected
                        if self.show_blunders_for_selected:
                            info = classify_destinations_for_selected_piece(
                                self.game_engine, self.legal_moves_for_selected
                            )
                            self.blunder_red_dests = info["red_dests"]
                            self.blunder_orange_dests = info["orange_dests"]
                            self.blunder_red_arrows = info["red_arrows"]
                            self.blunder_orange_arrows = info["orange_arrows"]
                        else:
                            self._clear_blunder_overlay()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                self.handle_click(mx, my)

    def _reset_game(self):
        self.game_engine.reset()
        self._clear_selection_and_overlays()
        self.red_captures = []
        self.yellow_captures = []
        self.half_move_count = 0
        self.board_flipped = False
        self.show_all_moves_red = False
        self.show_all_moves_yellow = False
        self.show_capturable_pieces = False
        self.best_move = None
        self.best_overall = None

    def _clear_blunder_overlay(self):
        self.blunder_red_dests = set()
        self.blunder_orange_dests = set()
        self.blunder_red_arrows = set()
        self.blunder_orange_arrows = set()

    def _clear_selection_and_overlays(self):
        self.selected_square = None
        self.legal_moves_for_selected = []
        self._clear_blunder_overlay()
        self.show_blunders_for_selected = False

    # --- Mouse logic ---
    def handle_click(self, mx: int, my: int):
        if self._try_click_homestone_icon(mx, my):
            return
        if not (self.board_left <= mx < self.board_left + self.board_draw_width and
                self.board_top  <= my < self.board_top  + self.board_draw_height):
            return

        board_x = mx - self.board_left
        board_y = my - self.board_top
        disp_col = int(board_x // self.square_size)
        disp_row = int(board_y // self.square_size)
        r, c = self._display_to_logical(disp_row, disp_col)

        if self.selected_square is None:
            state = self.game_engine.get_state()
            if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
                cell = state[r][c]
                if cell is None:
                    return
                _ptype, is_yellow_color, _has_moved = cell
                if is_yellow_color == self.game_engine.is_yellow_turn:
                    self.selected_square = (r, c)
                    all_moves = self.game_engine.get_legal_moves()
                    self.legal_moves_for_selected = [mv for mv in all_moves if mv[0] == (r, c)]
                    self._clear_blunder_overlay()
                    self.show_blunders_for_selected = False
                    self.best_move = None
        else:
            if self.selected_square == (r, c):
                self._clear_selection_and_overlays()
                self.best_move = None
                return

            possible_moves = [mv for mv in self.legal_moves_for_selected if mv[1] == (r, c)]
            if not possible_moves:
                self._clear_selection_and_overlays()
                self.best_move = None
                return

            chosen_move = possible_moves[0] if len(possible_moves) == 1 else self._prompt_promotion_choice(possible_moves)

            if chosen_move:
                state = self.game_engine.get_state()
                target_cell = state[r][c]
                if target_cell is not None:
                    captured_ptype, captured_is_yellow, _ = target_cell
                    if captured_is_yellow:
                        self.yellow_captures.append(captured_ptype)
                    else:
                        self.red_captures.append(captured_ptype)

                self.game_engine.apply_move(chosen_move)
                self.half_move_count += 1
                self.best_move = None
                self.best_overall = None
                self._clear_selection_and_overlays()

    # --- Promotion dialog ---
    def _prompt_promotion_choice(self, possible_moves):
        w = 300
        h = 150
        popup_rect = pygame.Rect(
            self.window_width//2 - w//2,
            self.window_height//2 - h//2,
            w, h
        )
        popup_surf = pygame.Surface((w, h))
        popup_surf.fill((200, 200, 200))

        font = pygame.font.SysFont(None, 26)
        txt = font.render("Choose promotion:", True, (0,0,0))
        popup_surf.blit(txt, (10,10))

        promos = []
        for mv in possible_moves:
            if len(mv) == 2:
                if None not in promos:
                    promos.append(None)
            elif len(mv) == 3:
                if mv[2] not in promos:
                    promos.append(mv[2])

        button_w = 80
        button_h = 40
        x_start = (w - (len(promos) * (button_w + 10))) // 2
        y_btn = 60
        buttons = []
        for i, pr in enumerate(promos):
            bx = x_start + i*(button_w + 10)
            by = y_btn
            rect = pygame.Rect(bx, by, button_w, button_h)
            label = "No Promo" if pr is None else ("Rider" if pr == RIDER else ("Tarnsman" if pr == TARNSMAN else f"Promo {pr}"))
            buttons.append((rect, pr, label))

        while True:
            self.draw()
            self.screen.blit(popup_surf, (popup_rect.x, popup_rect.y))
            for (brect, pr, label) in buttons:
                br = brect.move(popup_rect.x, popup_rect.y)
                pygame.draw.rect(self.screen, (180,180,180), br)
                pygame.draw.rect(self.screen, (0,0,0), br, 2)
                surf_txt = font.render(label, True, (0,0,0))
                lx = br.centerx - surf_txt.get_width()//2
                ly = br.centery - surf_txt.get_height()//2
                self.screen.blit(surf_txt, (lx, ly))

            pygame.display.flip()
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    self.running = False
                    return None
                elif evt.type == pygame.KEYDOWN:
                    if evt.key in (pygame.K_ESCAPE, pygame.K_q):
                        return None
                elif evt.type == pygame.MOUSEBUTTONDOWN and evt.button == 1:
                    mx, my = evt.pos
                    for (brect, pr, label) in buttons:
                        br = brect.move(popup_rect.x, popup_rect.y)
                        if br.collidepoint(mx, my):
                            for mv in possible_moves:
                                if len(mv) == 2 and pr is None:
                                    return mv
                                elif len(mv) == 3 and mv[2] == pr:
                                    return mv
                            return None

    # --- Homestone icons ---
    def _try_click_homestone_icon(self, mx, my):
        icon_size = 60
        red_rect = pygame.Rect(0, self._top_hud_h(), self.left_side_width, icon_size)
        yel_rect = pygame.Rect(self.window_width - self.right_side_width, self._top_hud_h(),
                               self.right_side_width, icon_size)
        if red_rect.collidepoint(mx, my):
            if not self.game_engine.red_homestone_placed:
                if self.selected_square == (-1, -1, "red"):
                    self._clear_selection_and_overlays()
                else:
                    self._select_unplaced_homestone(False)
                    self.selected_square = (-1, -1, "red")
            return True
        if yel_rect.collidepoint(mx, my):
            if not self.game_engine.yellow_homestone_placed:
                if self.selected_square == (-1, -1, "yellow"):
                    self._clear_selection_and_overlays()
                else:
                    self._select_unplaced_homestone(True)
                    self.selected_square = (-1, -1, "yellow")
            return True
        return False

    def _select_unplaced_homestone(self, is_yellow):
        all_moves = self.game_engine.get_legal_moves()
        self.legal_moves_for_selected = [mv for mv in all_moves if mv[0] == (-1, -1)]
        self._clear_blunder_overlay()

    # --- Coords helpers ---
    def _display_to_logical(self, rr, cc):
        if not self.board_flipped:
            return rr, cc
        else:
            return BOARD_ROWS - 1 - rr, BOARD_COLS - 1 - cc

    def _logical_to_display(self, rr, cc):
        if not self.board_flipped:
            return rr, cc
        else:
            return BOARD_ROWS - 1 - rr, BOARD_COLS - 1 - cc

    # --- Drawing ---
    def draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        self._draw_hud()
        self._draw_side_panels()
        self._draw_board_squares()
        self._draw_labels()
        self._draw_show_all_moves()
        self._draw_capturable_pieces()   # draws capture arrows when C is ON
        self._draw_blunder_overlays()    # blunder borders + arrows when E is ON
        self._draw_pieces()
        self._draw_captures()
        self._draw_homestone_icons()

        # Best overall move (press 'S')
        if self.best_overall is not None:
            start, end, *rest = self.best_overall
            start_r, start_c = start
            end_r, end_c = end
            disp_start = self._logical_to_display(start_r, start_c)
            disp_end   = self._logical_to_display(end_r, end_c)
            start_rect = pygame.Rect(
                self.board_left + disp_start[1] * self.square_size,
                self.board_top  + disp_start[0] * self.square_size,
                self.square_size, self.square_size
            )
            end_rect = pygame.Rect(
                self.board_left + disp_end[1] * self.square_size,
                self.board_top  + disp_end[0] * self.square_size,
                self.square_size, self.square_size
            )
            pygame.draw.rect(self.screen, (0,255,0), start_rect, 4)
            pygame.draw.rect(self.screen, (0,255,0), end_rect, 4)

    def _draw_hud(self):
        rect = pygame.Rect(0, 0, self.window_width, self._top_hud_h())
        pygame.draw.rect(self.screen, COLOR_HUD_BG, rect)
        turn = "Yellow" if self.game_engine.is_yellow_turn else "Red"
        mv_num = (self.half_move_count // 2) + 1
        if not self.game_engine.is_game_over():
            text = (
                "Move {} - {}'s turn "
                "(N=New, Q/Esc=Quit, F=Flip, R=ShowRed, Y=ShowYellow, "
                "C=Capturable, S=StrongHint, E=BlunderCheck)"
            ).format(mv_num, turn)
        else:
            winner = self.game_engine.get_winner()
            if winner is None:
                text = f"Move {mv_num} - Game Over! No winner? (N=New, Q/Esc=Quit)"
            else:
                text = f"Move {mv_num} - Game Over! {winner} wins! (N=New, Q/Esc=Quit)"
        surf = self.font_medium.render(text, True, COLOR_HUD_TEXT)
        self.screen.blit(surf, (10, 10))

    def _top_hud_h(self):
        return int(self.window_height * TOP_HUD_RATIO)

    def _draw_side_panels(self):
        left_rect  = pygame.Rect(0, self._top_hud_h(), self.left_side_width, self.window_height - self._top_hud_h())
        right_rect = pygame.Rect(self.window_width - self.right_side_width, self._top_hud_h(),
                                 self.right_side_width, self.window_height - self._top_hud_h())
        pygame.draw.rect(self.screen, (60,60,60), left_rect)
        pygame.draw.rect(self.screen, (60,60,60), right_rect)

    def _draw_board_squares(self):
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                dr, dc = self._logical_to_display(r, c)
                color = COLOR_LIGHT if ((r+c) % 2 == 0) else COLOR_DARK
                if self.selected_square is not None:
                    possible_ends = [m[1] for m in self.legal_moves_for_selected]
                    if (r, c) in possible_ends:
                        color = COLOR_HIGHLIGHT
                if self.selected_square == (r, c):
                    color = COLOR_SELECTED
                x = self.board_left + dc*self.square_size
                y = self.board_top + dr*self.square_size
                rect = pygame.Rect(x, y, self.square_size, self.square_size)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_labels(self):
        for rr in range(BOARD_ROWS):
            disp_r, _ = self._logical_to_display(rr, 0)
            lbl = str(rr)
            srf = self.font_small.render(lbl, True, (255,255,255))
            sy = self.board_top + disp_r*self.square_size + self.square_size//2 - srf.get_height()//2
            sx = self.board_left - srf.get_width() - 5
            self.screen.blit(srf, (sx, sy))
        for cc in range(BOARD_COLS):
            _, disp_c = self._logical_to_display(0, cc)
            lbl = str(cc)
            srf = self.font_small.render(lbl, True, (255,255,255))
            sx = self.board_left + disp_c*self.square_size + self.square_size//2 - srf.get_width()//2
            sy = self.board_top - srf.get_height() - 5
            self.screen.blit(srf, (sx, sy))

    def _draw_show_all_moves(self):
        if not (self.show_all_moves_red or self.show_all_moves_yellow):
            return

        red_targets = {}
        yellow_targets = {}
        original_turn = self.game_engine.is_yellow_turn

        if self.show_all_moves_red:
            self.game_engine.is_yellow_turn = False
            for mv in self.game_engine.get_legal_moves():
                end = mv[1]
                red_targets[end] = red_targets.get(end, 0) + 1
            red_protected = get_protected_squares(self.game_engine.board, color_flag=False)
            for idx, count in red_protected.items():
                rc = index_to_rc(idx)
                red_targets[rc] = red_targets.get(rc, 0) + count

        if self.show_all_moves_yellow:
            self.game_engine.is_yellow_turn = True
            for mv in self.game_engine.get_legal_moves():
                end = mv[1]
                yellow_targets[end] = yellow_targets.get(end, 0) + 1
            yellow_protected = get_protected_squares(self.game_engine.board, color_flag=True)
            for idx, count in yellow_protected.items():
                rc = index_to_rc(idx)
                yellow_targets[rc] = yellow_targets.get(rc, 0) + count

        self.game_engine.is_yellow_turn = original_turn
        self._draw_moves_overlay(red_targets, yellow_targets)

    def _draw_moves_overlay(self, red_targets, yellow_targets):
        font = pygame.font.SysFont(None, 16)
        squares = set(red_targets.keys()).union(yellow_targets.keys())
        for (r, c) in squares:
            dr, dc = self._logical_to_display(r, c)
            x = self.board_left + dc * self.square_size
            y = self.board_top + dr * self.square_size
            circle_radius = max(5, self.square_size // 10)
            if (r, c) in red_targets:
                count = red_targets[(r, c)]
                cx = x + 6
                cy = y + 6
                pygame.draw.circle(self.screen, COLOR_SHOW_RED, (cx, cy), circle_radius)
                label = f"x{count}"
                text = font.render(label, True, (0, 0, 0))
                self.screen.blit(text, (cx + circle_radius + 2, cy - text.get_height() // 2))
            if (r, c) in yellow_targets:
                count = yellow_targets[(r, c)]
                cx = x + self.square_size - 6 - circle_radius * 2
                cy = y + 6
                pygame.draw.circle(self.screen, COLOR_SHOW_YELLOW, (cx, cy), circle_radius)
                label = f"x{count}"
                text = font.render(label, True, (0, 0, 0))
                self.screen.blit(text, (cx + circle_radius + 2, cy - text.get_height() // 2))

    # ---------- CAPTURABLE OVERLAY (C) ----------
    def _draw_capturable_pieces(self):
        if not self.show_capturable_pieces:
            return

        squares_capturable = set()
        red_arrows = set()     # red attacker -> yellow victim
        yellow_arrows = set()  # yellow attacker -> red victim

        original_turn = self.game_engine.is_yellow_turn

        for color in [False, True]:  # False=Red to move, True=Yellow to move
            self.game_engine.is_yellow_turn = color
            st = self.game_engine.get_state()
            for mv in self.game_engine.get_legal_moves():
                er, ec = mv[1]
                if 0 <= er < BOARD_ROWS and 0 <= ec < BOARD_COLS:
                    occ = st[er][ec]
                    if occ is not None:
                        _ptype, is_yellow, _moved = occ
                        if is_yellow != color:  # capture of the opposite color
                            squares_capturable.add((er, ec))
                            if color:  # Yellow attacker
                                yellow_arrows.add((mv[0], mv[1]))
                            else:      # Red attacker
                                red_arrows.add((mv[0], mv[1]))

        self.game_engine.is_yellow_turn = original_turn

        # Red overlay on all capturable squares (existing behavior)
        for (rr, cc) in squares_capturable:
            dr, dc = self._logical_to_display(rr, cc)
            x = self.board_left + dc*self.square_size
            y = self.board_top  + dr*self.square_size
            overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            overlay.fill((*COLOR_ALL_CAPTURES, 120))
            self.screen.blit(overlay, (x, y))

        # NEW: arrows showing attackers -> victims for both sides
        red_width = max(3, self.square_size // 14)
        yel_width = max(3, self.square_size // 14)
        for (fr, fc), (tr, tc) in red_arrows:
            self._draw_arrow((fr, fc), (tr, tc), COLOR_SHOW_RED, width=red_width)
        for (fr, fc), (tr, tc) in yellow_arrows:
            self._draw_arrow((fr, fc), (tr, tc), COLOR_SHOW_YELLOW, width=yel_width)

    # ---------- BLUNDER OVERLAYS (E) ----------
    def _draw_blunder_overlays(self):
        if not self.show_blunders_for_selected:
            return
        thickness = max(3, self.square_size // 12)

        # Borders on destination squares
        for (rr, cc) in self.blunder_red_dests:
            dr, dc = self._logical_to_display(rr, cc)
            x = self.board_left + dc * self.square_size
            y = self.board_top  + dr * self.square_size
            pygame.draw.rect(self.screen, COLOR_BLUNDER_RED, pygame.Rect(x, y, self.square_size, self.square_size), thickness)

        for (rr, cc) in self.blunder_orange_dests:
            dr, dc = self._logical_to_display(rr, cc)
            x = self.board_left + dc * self.square_size
            y = self.board_top  + dr * self.square_size
            pygame.draw.rect(self.screen, COLOR_BLUNDER_ORANGE, pygame.Rect(x, y, self.square_size, self.square_size), thickness)

        # Arrows from enemy pieces to threatened squares
        for (fr, fc), (tr, tc) in self.blunder_red_arrows:
            self._draw_arrow((fr, fc), (tr, tc), COLOR_BLUNDER_RED, width=max(3, self.square_size // 14))
        for (fr, fc), (tr, tc) in self.blunder_orange_arrows:
            self._draw_arrow((fr, fc), (tr, tc), COLOR_BLUNDER_ORANGE, width=max(2, self.square_size // 16))

    # ---------- Arrow utilities ----------
    def _square_center_px(self, rr, cc):
        dr, dc = self._logical_to_display(rr, cc)
        cx = self.board_left + dc * self.square_size + self.square_size // 2
        cy = self.board_top  + dr * self.square_size + self.square_size // 2
        return cx, cy

    def _draw_arrow(self, from_sq, to_sq, color, width=2):
        """Draw a line with a small triangular arrowhead."""
        fx, fy = self._square_center_px(*from_sq)
        tx, ty = self._square_center_px(*to_sq)

        head_len = max(10, self.square_size // 4)
        vx, vy = tx - fx, ty - fy
        dist = math.hypot(vx, vy)
        if dist < 1e-3:
            return
        ux, uy = vx / dist, vy / dist
        bx, by = tx - ux * head_len, ty - uy * head_len

        # shaft
        pygame.draw.line(self.screen, color, (fx, fy), (bx, by), width)

        # arrowhead triangle
        perp = (-uy, ux)
        half = head_len * 0.4
        p1 = (tx, ty)
        p2 = (bx + perp[0] * half, by + perp[1] * half)
        p3 = (bx - perp[0] * half, by - perp[1] * half)
        pygame.draw.polygon(self.screen, color, [p1, p2, p3])

    def _draw_pieces(self):
        st = self.game_engine.get_state()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell = st[r][c]
                if cell:
                    disp_r, disp_c = self._logical_to_display(r, c)
                    ptype, is_yellow_color, _has_moved = cell
                    center_x = self.board_left + disp_c*self.square_size + self.square_size//2
                    center_y = self.board_top  + disp_r*self.square_size + self.square_size//2
                    radius = max(5, (self.square_size // 2) - 10)
                    fill_color = COLOR_YELLOW_PIECE if is_yellow_color else COLOR_RED_PIECE
                    pygame.draw.circle(self.screen, fill_color, (center_x, center_y), radius)
                    pygame.draw.circle(self.screen, CIRCLE_BORDER_COLOR, (center_x, center_y), radius, 2)
                    piece_img = self.piece_images[ptype]
                    if piece_img is not None:
                        side = int(radius * 1.4)
                        scaled_img = pygame.transform.smoothscale(piece_img, (side, side))
                        img_rect = scaled_img.get_rect(center=(center_x, center_y))
                        self.screen.blit(scaled_img, img_rect)
                    else:
                        symbol = PIECE_SYMBOLS.get((ptype, is_yellow_color), "?")
                        text_surf = self.font_medium.render(symbol, True, COLOR_TEXT)
                        text_rect = text_surf.get_rect(center=(center_x, center_y))
                        self.screen.blit(text_surf, text_rect)

    def _draw_captures(self):
        icon_size = min(self.left_side_width, 40)
        x_left = (self.left_side_width - icon_size) // 2
        for i, pt in enumerate(self.red_captures):
            y_offset = self._top_hud_h() + 70 + i*icon_size
            self._draw_capture_icon(pt, False, x_left, y_offset, icon_size)
        x_right = self.window_width - self.right_side_width + (self.right_side_width - icon_size)//2
        for i, pt in enumerate(self.yellow_captures):
            y_offset = self._top_hud_h() + 70 + i*icon_size
            self._draw_capture_icon(pt, True, x_right, y_offset, icon_size)

    def _draw_capture_icon(self, ptype, is_yellow_color, x, y, size):
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, (80,80,80), rect)
        center = (x + size//2, y + size//2)
        radius = (size//2) - 3
        fill_color = COLOR_YELLOW_PIECE if is_yellow_color else COLOR_RED_PIECE
        pygame.draw.circle(self.screen, fill_color, center, radius)
        pygame.draw.circle(self.screen, (0,0,0), center, radius, 2)
        piece_img = self.piece_images[ptype]
        if piece_img is not None:
            side = int(radius * 1.4)
            scaled_img = pygame.transform.smoothscale(piece_img, (side, side))
            img_rect = scaled_img.get_rect(center=center)
            self.screen.blit(scaled_img, img_rect)
        else:
            symbol = PIECE_SYMBOLS.get((ptype, is_yellow_color), "?")
            text_surf = self.font_medium.render(symbol, True, COLOR_TEXT)
            text_rect = text_surf.get_rect(center=center)
            self.screen.blit(text_surf, text_rect)

    def _draw_homestone_icons(self):
        icon_size = 60
        red_rect = pygame.Rect(0, self._top_hud_h(), self.left_side_width, icon_size)
        pygame.draw.rect(self.screen, (60,60,60), red_rect)
        if not self.game_engine.red_homestone_placed:
            cx = red_rect.x + red_rect.width//2
            cy = red_rect.y + red_rect.height//2
            radius = (icon_size//2) - 5
            pygame.draw.circle(self.screen, COLOR_RED_PIECE, (cx, cy), radius)
            pygame.draw.circle(self.screen, (0,0,0), (cx, cy), radius, 2)
            piece_img = self.piece_images[HOMESTONE]
            if piece_img:
                side = int(radius * 1.4)
                scaled = pygame.transform.smoothscale(piece_img, (side, side))
                img_rect = scaled.get_rect(center=(cx, cy))
                self.screen.blit(scaled, img_rect)
            else:
                symbol = PIECE_SYMBOLS.get((HOMESTONE, False), "?")
                srf = self.font_medium.render(symbol, True, COLOR_TEXT)
                rect = srf.get_rect(center=(cx, cy))
                self.screen.blit(srf, rect)

        yel_rect = pygame.Rect(self.window_width - self.right_side_width, self._top_hud_h(),
                               self.right_side_width, icon_size)
        pygame.draw.rect(self.screen, (60,60,60), yel_rect)
        if not self.game_engine.yellow_homestone_placed:
            cx = yel_rect.x + yel_rect.width//2
            cy = yel_rect.y + yel_rect.height//2
            radius = (icon_size//2) - 5
            pygame.draw.circle(self.screen, COLOR_YELLOW_PIECE, (cx, cy), radius)
            pygame.draw.circle(self.screen, (0,0,0), (cx, cy), radius, 2)
            piece_img = self.piece_images[HOMESTONE]
            if piece_img:
                side = int(radius * 1.4)
                scaled = pygame.transform.smoothscale(piece_img, (side, side))
                img_rect = scaled.get_rect(center=(cx, cy))
                self.screen.blit(scaled, img_rect)
            else:
                symbol = PIECE_SYMBOLS.get((HOMESTONE, True), "?")
                srf = self.font_medium.render(symbol, True, COLOR_TEXT)
                rect = srf.get_rect(center=(cx, cy))
                self.screen.blit(srf, rect)

# Run
if __name__ == "__main__":
    gui = KaissaGUI()
    gui.run()
