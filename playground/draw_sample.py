import pygame
import sys
import math

# 初始化Pygame
pygame.init()

# 设置画布大小
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Circle Drawer")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # 用于选中状态

# 圆形列表：每个圆是一个字典 {center: (x, y), radius: r}
circles = []

# 状态变量
drawing = False  # 是否正在绘制新圆
selected_circle = None  # 当前选中的圆
resizing = False  # 是否正在调整大小
moving = False  # 是否正在移动圆
move_offset = (0, 0)  # 移动时的偏移

def draw_circles():
    screen.fill(WHITE)
    for circle in circles:
        color = RED if circle == selected_circle else BLACK
        pygame.draw.circle(screen, color, circle['center'], circle['radius'], 2)  # 绘制轮廓，便于查看

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_circle_at_pos(pos):
    for circle in circles:
        dist = distance(pos, circle['center'])
        if abs(dist - circle['radius']) < 5 or dist < circle['radius']:  # 点击圆内或边缘附近
            return circle
    return None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键
                pos = pygame.mouse.get_pos()
                clicked_circle = find_circle_at_pos(pos)
                if clicked_circle:
                    selected_circle = clicked_circle
                    if resizing:
                        # 如果已经在调整大小模式，忽略（这里简化，假设先选后调整）
                        pass
                    else:
                        # 开始移动
                        moving = True
                        move_offset = (clicked_circle['center'][0] - pos[0], clicked_circle['center'][1] - pos[1])
                else:
                    # 开始绘制新圆
                    drawing = True
                    new_center = pos
                    circles.append({'center': new_center, 'radius': 0})
                    selected_circle = circles[-1]
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pos = pygame.mouse.get_pos()
                selected_circle['radius'] = int(distance(new_center, pos))
            elif moving and selected_circle:
                # 移动圆
                pos = pygame.mouse.get_pos()
                selected_circle['center'] = (pos[0] + move_offset[0], pos[1] + move_offset[1])
            elif resizing and selected_circle:
                # 调整大小：从中心到鼠标的距离作为新半径
                pos = pygame.mouse.get_pos()
                selected_circle['radius'] = int(distance(selected_circle['center'], pos))
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                moving = False  # 停止移动
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # 按R切换调整大小模式
                resizing = not resizing
            elif event.key == pygame.K_DELETE and selected_circle:  # 删除选中圆
                circles.remove(selected_circle)
                selected_circle = None
            elif event.key == pygame.K_RETURN:  # 按Enter输出代码
                print("\ndef gen_sample():")
                print("    image = np.ones((400, 400, 3), dtype=np.uint8) * 255")
                for circle in circles:
                    if circle['radius'] > 0:  # 只输出半径大于0的圆
                        print(f"    cv2.circle(image, {circle['center']}, {circle['radius']}, (0, 0, 0), -1)")
                print("    return image\n")
        elif event.type == pygame.MOUSEWHEEL and selected_circle and resizing:
            # 滚轮调整大小（备用，如果鼠标拖拽不方便）
            selected_circle['radius'] += event.y * 5
            selected_circle['radius'] = max(1, selected_circle['radius'])

    draw_circles()
    if resizing:
        font = pygame.font.SysFont(None, 24)
        text = font.render("Resizing Mode", True, RED)
        screen.blit(text, (10, 10))
    pygame.display.flip()

pygame.quit()
sys.exit()