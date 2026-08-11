import streamlit as st
from inference import WalletRiskModel


st.set_page_config(
    page_title="Wallet Risk Scoring",
    page_icon="🔎",
    layout="centered",
)


@st.cache_resource
def load_model():

    return WalletRiskModel()


model = load_model()


st.title("🔎 Wallet Risk Scoring")

st.write(
    "GraphSAGE-based Ethereum wallet risk analysis."
)

st.divider()


address = st.text_input(
    "Wallet address",
    placeholder="0x...",
)


if st.button(
    "Analyze Wallet",
    use_container_width=True,
):

    if not address:

        st.warning(
            "Please enter a wallet address."
        )

    else:

        with st.spinner(
            "Analyzing wallet..."
        ):

            result = model.predict(
                address
            )

        if result is None:

            st.error(
                "Wallet not found in the current "
                "Stage 1 dataset."
            )

            st.info(
                "Stage 1 currently supports wallets "
                "present in the processed dataset."
            )

        else:

            score = result["risk_score"]

            st.divider()

            if score >= 0.70:

                st.error(
                    f"HIGH RISK — {score:.2%}"
                )

            elif score >= 0.30:

                st.warning(
                    f"MEDIUM RISK — {score:.2%}"
                )

            else:

                st.success(
                    f"LOW RISK — {score:.2%}"
                )

            st.subheader(
                "Prediction"
            )

            if result["prediction"] == 1:

                st.error(
                    "Malicious"
                )

            else:

                st.success(
                    "Benign"
                )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Risk Score",
                    f"{score:.2%}",
                )

                st.metric(
                    "Node ID",
                    result["node_id"],
                )

            with col2:

                st.metric(
                    "Known Label",
                    (
                        "Malicious"
                        if result["label"] == 1
                        else "Benign"
                    ),
                )

                st.write(
                    f"**Type:** {result['type']}"
                )

            st.divider()

            st.subheader(
                "Wallet Address"
            )

            st.code(
                result["address"]
            )

            st.caption(
                "Risk score is the model's malicious-class "
                "score and is not a calibrated probability."
            )