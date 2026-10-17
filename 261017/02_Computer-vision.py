# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
ball_mask_final = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    ball_mask_closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
# TODO: 최종 농구공 영역 및 외곽선 선택
ball_img = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)

    ball_region = np.zeros_like(ball_mask_final)
    cv2.drawContours(ball_region, [largest_contour], -1, 255, thickness=cv2.FILLED)

    mask_color = np.zeros_like(ball_img)
    mask_color[:, :, 1] = ball_region
    ball_img = cv2.addWeighted(ball_img, 1.0, mask_color, 0.35, 0)
    cv2.drawContours(ball_img, [largest_contour], -1, (0,255,0), 2)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(mask_enhanced,cmap="gray"), plt.title("Enhanced Mask")
plt.subplot(1,3,2), plt.imshow(ball_mask_final, cmap="gray"), plt.title("Mask (open x2, close x2)")
plt.subplot(1,3,3), plt.imshow(ball_img), plt.title("Largest Contour")

for ax in plt.gcf().axes:
    ax.axis("off")