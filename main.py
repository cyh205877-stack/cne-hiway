import base64
import mimetypes
from datetime import datetime

import streamlit as st
from openai import OpenAI


# -----------------------------
# Utils
# -----------------------------
def get_client() -> OpenAI:
    """
    Streamlit secrets에서 키를 읽어 OpenAI 클라이언트를 생성합니다.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error('st.secrets["OPENAI_API_KEY"]가 설정되어 있지 않습니다. (Streamlit Cloud > Settings > Secrets)')
        st.stop()
    return OpenAI(api_key=api_key)


def file_to_data_url(uploaded_file) -> str:
    """
    업로드된 파일(bytes)을 data URL로 변환합니다.
    (예: data:image/png;base64,....)
    """
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type

    # 일부 환경에서 type이 비어있을 수 있어 보정
    if not mime:
        guessed, _ = mimetypes.guess_type(uploaded_file.name)
        mime = guessed or "application/octet-stream"

    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_prompt(user_profile: dict) -> str:
    """
    봄 여행 추천 보고서 생성을 위한 프롬프트(한국어).
    """
    style = user_profile["style"]
    budget = user_profile["budget"]
    region = user_profile["region"]
    travel_days = user_profile["travel_days"]
    with_who = user_profile["with_who"]
    interests = user_profile["interests"]
    constraints = user_profile["constraints"]

    return f"""
당신은 여행 기획 전문가입니다. 사용자는 '봄에 여행 가기 좋은 곳'을 찾고 있습니다.
아래 조건을 반영해, 한국어로 읽기 쉬운 '여행 추천 보고서'를 작성하세요.

[사용자 조건]
- 여행 지역 선호: {region}
- 여행 기간: {travel_days}일
- 동행: {with_who}
- 예산 감각: {budget}
- 관심사: {interests}
- 제약/요청사항: {constraints}
- 글 스타일: {style}

[출력 형식]
1) 요약(3~5줄)
2) 추천 후보 TOP 5 (각 후보: 한줄 매력 포인트 + 왜 봄에 좋은지 + 예상 체감 비용(낮음/보통/높음))
3) TOP 1 상세 플랜
   - 추천 시기(예: 3월~5월 중 언제가 좋은지)
   - 1일차~{travel_days}일차 일정(오전/오후/저녁)
   - 이동 팁(대중교통/렌터카/도보 관점)
   - 맛집/시장/로컬 체험 아이디어(과도한 단정 금지)
4) 비가 오거나 미세먼지가 심할 때 대체 일정 5개
5) 준비물 체크리스트(봄 날씨 변동 고려)
6) (선택) 업로드된 이미지가 있다면, 이미지에서 보이는 분위기/풍경/활동을 5줄로 해석하고,
   그 취향에 맞게 추천을 미세 조정했다는 코멘트를 2~3줄 추가

[주의]
- 특정 사실(축제 일정/가격 등)을 단정적으로 고정하지 말고, "가능성이 높다/일반적으로/현지에 따라 다를 수 있다" 식으로 표현하세요.
- 과도한 광고 문구 금지.
""".strip()


def make_report(client: OpenAI, prompt_text: str, image_data_url: str | None) -> str:
    """
    OpenAI Responses API를 사용해 보고서를 생성합니다.
    """
    model = "gpt-4o-mini"  # 멀티모달(이미지 포함) 가능 모델

    # Responses API 입력 구성
    if image_data_url:
        user_content = [
            {"type": "input_text", "text": prompt_text},
            {"type": "input_text", "text": "아래 이미지를 참고하여 여행 취향/분위기를 추론해 추천을 미세 조정하세요."},
            {"type": "input_image", "image_url": image_data_url},
        ]
    else:
        user_content = [{"type": "input_text", "text": prompt_text}]

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
    )

    # SDK가 제공하는 편의 프로퍼티(output_text)를 우선 사용
    text = getattr(resp, "output_text", None)
    if text:
        return text

    # 혹시 모를 케이스(구조가 달라도 안전하게)
    try:
        chunks = []
        for item in resp.output:
            for c in item.content:
                if c.type in ("output_text", "text"):
                    chunks.append(getattr(c, "text", ""))
        return "\n".join([t for t in chunks if t]).strip()
    except Exception:
        return "보고서 생성에 실패했습니다. (응답 파싱 오류)"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="봄 여행 추천 보고서",
    page_icon="🌸",
    layout="wide",
)

st.title("🌸 봄에 여행 가기 좋은 곳 — 이미지 기반 추천 보고서")
st.caption("이미지를 올리면 분위기/취향을 참고해 봄 여행지를 추천하고, 일정까지 보고서로 만들어 드립니다.")

with st.sidebar:
    st.header("🧭 옵션 설정")
    region = st.selectbox(
        "여행 지역 선호",
        ["상관없음", "국내", "일본", "동남아", "유럽", "미주", "오세아니아", "기타"],
        index=0,
    )
    travel_days = st.slider("여행 기간(일)", min_value=1, max_value=14, value=4)
    with_who = st.selectbox("동행", ["혼자", "연인", "친구", "가족", "동료/단체", "기타"], index=3)
    budget = st.selectbox("예산 감각", ["가성비", "보통", "여유롭게"], index=1)
    interests = st.multiselect(
        "관심사(복수 선택)",
        ["자연/풍경", "꽃/정원", "맛집/미식", "카페", "온천/휴식", "트레킹", "도시 감성", "사진", "아이 동반", "축제/이벤트", "쇼핑"],
        default=["자연/풍경", "꽃/정원", "맛집/미식"],
    )
    style = st.selectbox("보고서 스타일", ["깔끔한 보고서", "친근한 말투", "디테일 중시(길게)"], index=0)
    constraints = st.text_area(
        "제약/요청사항(선택)",
        placeholder="예: 비행시간 짧게 / 렌터카 없이 / 유모차 가능 / 알레르기 / 걷는 양 적게 등",
    )

st.subheader("1) 이미지 업로드(선택)")
uploaded = st.file_uploader("봄에 가고 싶은 분위기의 사진(풍경/거리/카페 등)을 올려주세요.", type=["png", "jpg", "jpeg", "webp"])

colA, colB = st.columns([1, 1])

with colA:
    if uploaded:
        st.image(uploaded, caption="업로드한 이미지", use_container_width=True)
    else:
        st.info("이미지는 선택입니다. 이미지를 올리면 취향을 더 잘 맞춰드립니다.")

with colB:
    st.subheader("2) 보고서 생성")
    st.write("설정한 옵션 + (선택) 이미지 분위기를 반영해 **봄 여행 추천 보고서**를 생성합니다.")
    generate = st.button("📝 보고서 만들기", type="primary", use_container_width=True)

if "report" not in st.session_state:
    st.session_state.report = ""
if "last_generated" not in st.session_state:
    st.session_state.last_generated = None

if generate:
    client = get_client()

    user_profile = {
        "region": region,
        "travel_days": travel_days,
        "with_who": with_who,
        "budget": budget,
        "interests": ", ".join(interests) if interests else "특별히 없음",
        "constraints": constraints.strip() if constraints.strip() else "없음",
        "style": style,
    }

    prompt_text = build_prompt(user_profile)
    image_data_url = file_to_data_url(uploaded) if uploaded else None

    with st.spinner("보고서를 생성 중입니다..."):
        try:
            report = make_report(client, prompt_text, image_data_url)
            st.session_state.report = report
            st.session_state.last_generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error("오류가 발생했습니다. 아래 내용을 확인해 주세요.")
            st.exception(e)

st.divider()
st.subheader("3) 결과 보고서")

if st.session_state.report:
    if st.session_state.last_generated:
        st.caption(f"마지막 생성 시각: {st.session_state.last_generated}")
    st.markdown(st.session_state.report)

    # 다운로드 버튼(텍스트)
    filename = f"spring_travel_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button(
        label="⬇️ 보고서 다운로드(.txt)",
        data=st.session_state.report,
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
    )
else:
    st.warning("아직 보고서가 없습니다. 위에서 **보고서 만들기**를 눌러주세요.")

st.divider()
with st.expander("🔧 배포 체크리스트(간단)"):
    st.markdown(
        """
- GitHub에 `main.py`, `requirements.txt` 업로드  
- Streamlit Community Cloud에서 앱 생성 시 **Main file path = `main.py`**  
- Streamlit Cloud > Settings > Secrets에 아래처럼 등록:
  ```toml
  OPENAI_API_KEY = "당신의_API_키"
