import pygame
from PIL import Image
from model import Net
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Initialize Pygame
pygame.init()

# Define the window size and title
window_width = 480
window_height = 480
window_title = "Handwritten Digit Recognition"

FPS = 100

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255,0,0)
# Create the Pygame window
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption(window_title)

# Set up the drawing surface
WIN = pygame.Surface((window_width, window_height))
WIN.fill(BLACK)


#load the model
model = Net()
device = torch.device('cpu')
model.load_state_dict(torch.load(r'.\model_state_dict.pth',map_location=device))
model.eval()

#Transformation pipeline for preprocessing the drawn image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def process_image(img):
    img = Image.fromarray(img).convert('L')
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def predict_digit(image):
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)
    predicted_digit = predicted_class.item()
    predicted_probability = probabilities[0, predicted_digit].item()
    return (predicted_digit, predicted_probability)

def draw_screen(WIN):
    screen.blit(WIN, (0, 0))
    pygame.display.update()

def drawing(WIN):
    pos = pygame.mouse.get_pos()
    pygame.draw.circle(WIN,WHITE, pos, 20)
    
def clearing(WIN):
    pos = pygame.mouse.get_pos()
    pygame.draw.circle(WIN, BLACK, pos, 20)

def draw_prediction(WIN, pred):
    
    digit, probability = pred

    
    font = pygame.font.Font(None, 36)

    
    digit_surface = font.render(f"Prediction: {digit}", True,RED )
    probability_surface = font.render(f"Probability: {probability:.2f}", True, RED)

    
    WIN.blit(digit_surface, (10, 10))
    WIN.blit(probability_surface, (10, 50))

    
running = True
is_drawing = False
is_clearing = False
while running:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and not is_clearing:
                is_drawing = True
            elif event.button == 3 and not is_drawing:
                is_clearing = True 
        elif event.type == pygame.MOUSEBUTTONUP :
            is_drawing = False
            is_clearing = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                    WIN.fill(BLACK)
                    
            if event.key == pygame.K_SPACE:
                img = pygame.surfarray.array2d(screen).transpose()
                img = process_image(img)
                pred = predict_digit(img)
                draw_prediction(WIN,pred)
                
    if is_drawing:
        drawing(WIN)
    if is_clearing:
        clearing(WIN)
        
    draw_screen(WIN)
    
pygame.quit()