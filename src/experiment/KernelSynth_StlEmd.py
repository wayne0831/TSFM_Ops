# =====================================================================
# 4. 실행 및 시각화 테스트 (25개 샘플 생성, DataFrame 구축, 5x5 Grid Plot)
# =====================================================================
if __name__ == "__main__":
    # 재현성을 위한 난수 생성기 시드 고정
    np.random.seed(42)

    # 31개 기본 커널 뱅크 인스턴스 빌드
    bank = build_kernel_bank()

    samples_data = []      # 생성된 시계열 수치 배열을 저장할 리스트
    metadata_records = []  # 커널 조합 메타데이터를 저장할 리스트
    num_samples = 49       # 8x8 그리드 생성을 위한 총 샘플 수

    # 64회 반복하며 가상 시계열 및 메타데이터 추출
    for i in range(1, num_samples + 1):
        ts, meta = kernel_synth(bank, sample_id=i, max_kernels=5, length=512)
        samples_data.append(ts)
        metadata_records.append(meta)

    # -----------------------------------------------------------------
    # (1) 조합 메타데이터를 Pandas DataFrame으로 변환 및 검증
    # -----------------------------------------------------------------
    df_kernels = pd.DataFrame(metadata_records)

    print("=== KernelSynth Generation Metadata DataFrame ===")
    # 주요 메타 컬럼(샘플ID, 커널수, 최종수식, 적용연산자) 상위 5개 출력
    print(df_kernels[["Sample_ID", "Num_Kernels", "Kernel_Expression", "Operations"]].head(100))
    print("\n전체 컬럼 목록:", list(df_kernels.columns))

    # -----------------------------------------------------------------
    # (2) 5x5 서브플롯 그리드 생성 및 Title에 커널 조합 수식 매핑
    # -----------------------------------------------------------------
    # 가로 20인치, 세로 12인치의 여유 있는 캔버스 생성 (X축 공유)
    fig, axes = plt.subplots(7, 7, figsize=(20, 12), sharex=True)
    axes = axes.flatten()  # 2차원 축 배열(5, 5)을 1차원(25,)으로 펼쳐 인덱싱 편의성 확보

    for idx in range(num_samples):
        ax = axes[idx]
        ts = samples_data[idx]
        expr = metadata_records[idx]["Kernel_Expression"]

        # 시계열 라인 플롯 렌더링
        ax.plot(ts, color="tab:blue", lw=1.1)

        # 서브플롯 타이틀에 샘플 번호와 축약된 커널 조합 수식을 함께 명시
        # 폭이 좁은 5x5 환경에서 텍스트 겹침을 방지하기 위해 폰트 크기 7.5pt 지정
        ax.set_title(f"#{idx+1}: {expr}", fontsize=7.5, fontweight="bold", pad=4)
        #ax.set_title(f"{expr}", fontsize=7.5, fontweight="bold", pad=4)

        # 가독성을 높이기 위한 반투명 보조선 및 눈금 폰트 조정
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # 서브플롯 간 여백 자동 조정 후 화면 출력
    plt.tight_layout()
    plt.show()