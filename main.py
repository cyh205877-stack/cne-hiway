import streamlit as st
import openai
from PIL import Image
import io

# OpenAI API 키 로딩
openai.api_key = st.secrets["OPENAI_API_KEY"]

def analyze_image(image: Image.Image):
    """이미지를 분석하여 봄 여행지에 대한 보고서를 생성하는 함수"""
    
    # 이미지 파일을 바이트로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # OpenAI의 이미지를 분석하는 API 호출 (예시: 이미지 설명 생성)
    try:
        response = openai.Image.create(
            prompt="봄에 여행 가기 좋은 장소를 묘사하는 이미지를 분석해 주세요.",
            n=1,
            size="1024x1024",
            image=img_byte_arr
        )
        return response['data'][0]['text']
    except Exception as e:
        return f"오류 발생: {e}"

def main():
    st.title("봄에 여행 가기 좋은 곳")
    st.write("이 웹앱은 이미지를 업로드하면 봄 여행지 추천을 분석하여 보고서를 출력합니다.")
    
    # 이미지 업로드
    uploaded_image = st.file_uploader("여행지 이미지를 업로드 해 주세요.", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # 이미지 표시
        image = Image.open(uploaded_image)
        st.image(image, caption="업로드된 이미지", use_column_width=True)

        # 분석 버튼
        if st.button("분석하기"):
            with st.spinner("이미지를 분석 중입니다..."):
                result = analyze_image(image)
                st.write("### 분석 결과:")
                st.write(result)

if __name__ == "__main__":
    main()
