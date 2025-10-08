import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QFileDialog, QLabel, QMessageBox, QComboBox, QSpinBox, QGroupBox, QGridLayout,
    QScrollArea
)
from PyQt5.QtCore import Qt


def _find_header_row(df_raw: pd.DataFrame, max_scan: int = 8) -> int:
    """
    Tìm dòng header hợp lệ trong df_raw (đọc với header=None).
    Tiêu chí:
      - Có ít nhất 2 ô không rỗng/NaN
      - Không phải toàn 'Unnamed'
    """
    n = min(max_scan, len(df_raw))
    for i in range(n):
        row = df_raw.iloc[i]
        # Ô không rỗng
        non_blank = row.map(lambda x: str(x).strip()).replace("nan", "").astype(bool).sum()
        # Không phải tất cả đều là Unnamed
        is_all_unnamed = all(
            str(v).strip().lower().startswith("unnamed") or str(v).strip() == "" or str(v) == "nan"
            for v in row
        )
        if non_blank >= 2 and not is_all_unnamed:
            return i
    return 0  # fallback


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Loại bỏ cột toàn NaN
    - Strip tên cột, thay 'Unnamed: x' bằng rỗng
    - Cố gắng đặt tên cột CR (Compression Ratio) là 'CR'
    """
    # Bỏ cột toàn NaN
    df = df.dropna(axis=1, how='all').copy()

    # Chuẩn hóa tên cột
    new_cols = []
    for c in df.columns:
        name = str(c).strip()
        if name.lower().startswith("unnamed"):
            name = ""
        new_cols.append(name)
    df.columns = new_cols

    # Nếu vẫn còn cột không tên -> tự gán tên tạm
    fixed_cols = []
    for idx, c in enumerate(df.columns):
        fixed_cols.append(c if c else f"col_{idx}")
    df.columns = fixed_cols

    # Tìm cột CR:
    # - Nếu có cột tên chứa 'CR' hay 'Compression' thì dùng
    # - Nếu không, chọn cột đầu tiên có kiểu số và có 3 giá trị phổ biến như 10/20/30 (không bắt buộc)
    candidates = [c for c in df.columns if "cr" in c.lower() or "compression" in c.lower()]
    cr_col = None
    if candidates:
        cr_col = candidates[0]
    else:
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 2:
                # coi đây là CR nếu có vẻ là cột số đầu tiên
                cr_col = c
                break

    if cr_col is not None and cr_col != "CR":
        df = df.rename(columns={cr_col: "CR"})

    return df


class ExcelChartTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Tool Tạo Biểu Đồ Từ Excel/CSV')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # --- Điều khiển ---
        control_group = QGroupBox("Điều khiển")
        control_layout = QHBoxLayout(control_group)

        self.upload_btn = QPushButton("Chọn File Excel/CSV")
        self.upload_btn.clicked.connect(self.upload_file)
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; border: none; border-radius: 5px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #45a049; }
        """)

        self.chart_btn = QPushButton("Tạo Biểu Đồ")
        self.chart_btn.clicked.connect(self.create_chart)
        self.chart_btn.setMinimumHeight(40)
        self.chart_btn.setEnabled(False)
        self.chart_btn.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; border: none; border-radius: 5px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)

        self.save_btn = QPushButton("Lưu Biểu Đồ")
        self.save_btn.clicked.connect(self.save_chart)
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton { background-color: #FF9800; color: white; border: none; border-radius: 5px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)

        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.chart_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addStretch()

        # --- Tùy chọn ---
        options_group = QGroupBox("Tùy chọn biểu đồ")
        options_layout = QGridLayout(options_group)

        options_layout.addWidget(QLabel("Loại biểu đồ:"), 0, 0)
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Bar Chart", "Line Chart", "Scatter Plot"])
        options_layout.addWidget(self.chart_type_combo, 0, 1)

        options_layout.addWidget(QLabel("Kích thước:"), 0, 2)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["10x6", "12x8", "14x10", "16x12"])
        self.size_combo.setCurrentText("12x8")
        options_layout.addWidget(self.size_combo, 0, 3)

        options_layout.addWidget(QLabel("DPI:"), 1, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 300)
        self.dpi_spin.setValue(100)
        options_layout.addWidget(self.dpi_spin, 1, 1)

        self.file_label = QLabel("Chưa chọn file")
        self.file_label.setStyleSheet("color: #666; font-style: italic;")

        # Canvas
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)

        main_layout.addWidget(control_group)
        main_layout.addWidget(options_group)
        main_layout.addWidget(self.file_label)
        main_layout.addWidget(scroll_area)

    # ================== Đọc file (đÃ SỬA) ==================
    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file Excel hoặc CSV",
            "",
            "Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)"
        )

        if not file_path:
            return

        try:
            # Đọc raw không header để dò dòng tiêu đề thật
            if file_path.endswith(('.xlsx', '.xls')):
                df_raw = pd.read_excel(file_path, header=None, dtype=object)
            elif file_path.endswith('.csv'):
                df_raw = pd.read_csv(file_path, header=None, dtype=object)
            else:
                QMessageBox.warning(self, "Lỗi", "Định dạng file không được hỗ trợ!")
                return

            header_row = _find_header_row(df_raw, max_scan=8)

            # Đọc lại với header đúng
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, header=header_row)
            else:
                df = pd.read_csv(file_path, header=header_row)

            # Làm sạch cột, chuẩn hoá CR
            df = _clean_columns(df)

            # Loại bỏ hàng/ cột rỗng
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')

            # Hiển thị thông tin & set state
            self.data = df
            self.file_label.setText(f"Đã chọn: {os.path.basename(file_path)}")
            self.chart_btn.setEnabled(True)

            info_text = (
                f"Đã đọc thành công! (header ở dòng: {header_row})\n"
                f"Số dòng: {len(self.data)}\nSố cột: {len(self.data.columns)}\n\n"
                f"Các cột:\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(self.data.columns)]) +
                f"\n\nDữ liệu mẫu (3 dòng đầu):\n{self.data.head(3).to_string(index=False)}"
            )
            QMessageBox.information(self, "Thông tin dữ liệu", info_text)

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể đọc file: {str(e)}")

    # ================== VẼ BIỂU ĐỒ ==================
    def create_chart(self):
        if self.data is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn file trước!")
            return

        try:
            self.figure.clear()

            size = self.size_combo.currentText().split('x')
            width, height = int(size[0]), int(size[1])
            self.figure.set_size_inches(width, height)
            self.figure.set_dpi(self.dpi_spin.value())

            ax = self.figure.add_subplot(111)

            df = self.data.copy()

            # Xác định cột CR
            cr_col = "CR" if "CR" in df.columns else None
            if cr_col is None:
                # fallback: cột số đầu tiên
                for c in df.columns:
                    if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2:
                        cr_col = c
                        break

            if cr_col is None:
                QMessageBox.warning(self, "Lỗi", "Không tìm thấy cột CR (Compression Ratio)!")
                return

            # Ép kiểu số cho CR và sort
            df[cr_col] = pd.to_numeric(df[cr_col], errors="coerce")
            df = df.dropna(subset=[cr_col]).sort_values(by=cr_col)
            cr_values = df[cr_col].unique().tolist()
            x_pos = np.arange(len(cr_values))

            # Các cột giá trị (loại bỏ CR)
            data_columns = [c for c in df.columns if c != cr_col]

            chart_type = self.chart_type_combo.currentText()
            width_bar = 0.85 / max(len(data_columns), 1)  # gói gọn trong cụm
            # màu + hatch chỉ để phân biệt (không bắt buộc)
            colors = ['#FFD700', '#8B4513', '#228B22', '#D3D3D3', '#FF8C00', '#90EE90', '#4169E1', '#F4A460', '#8A2BE2']
            patterns = ['///', '', '\\\\\\', '...', 'ooo', '+++', 'xxx', None, None]

            def series_for(col):
                vals = []
                for v in cr_values:
                    row = df[df[cr_col] == v]
                    if not row.empty and col in row.columns:
                        val = pd.to_numeric(row[col].iloc[0], errors="coerce")
                        vals.append(float(val) if pd.notna(val) else 0.0)
                    else:
                        vals.append(0.0)
                return vals

            if chart_type == "Bar Chart":
                for i, col in enumerate(data_columns):
                    vals = series_for(col)
                    bars = ax.bar(x_pos + i * width_bar - (len(data_columns)-1)*width_bar/2,
                                  vals, width_bar, label=col,
                                  color=colors[i % len(colors)], alpha=0.9,
                                  edgecolor='black', linewidth=0.4)
                    if patterns[i % len(patterns)]:
                        for b in bars:
                            b.set_hatch(patterns[i % len(patterns)])

            elif chart_type == "Line Chart":
                for i, col in enumerate(data_columns):
                    vals = series_for(col)
                    ax.plot(x_pos, vals, marker='o', linewidth=2, markersize=6,
                            label=col, color=colors[i % len(colors)])

            else:  # Scatter Plot
                for i, col in enumerate(data_columns):
                    vals = series_for(col)
                    ax.scatter(x_pos, vals, s=90, label=col,
                               color=colors[i % len(colors)], alpha=0.85, edgecolors='black')

            # Nhãn & định dạng
            ax.set_xlabel('Compression Ratio (CR)', fontsize=12, fontweight='bold')
            ax.set_ylabel('NC', fontsize=12, fontweight='bold')
            ax.set_title('So sánh hiệu suất các phương thức', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in cr_values])
            ax.set_ylim(0.5, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

            self.figure.tight_layout()
            self.canvas.draw()
            self.save_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo biểu đồ: {str(e)}")

    def save_chart(self):
        if not self.figure.get_axes():
            QMessageBox.warning(self, "Lỗi", "Không có biểu đồ để lưu!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu biểu đồ",
            "chart.png",
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)"
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=self.dpi_spin.value(),
                                    bbox_inches='tight', facecolor='white')
                QMessageBox.information(self, "Thành công", f"Đã lưu biểu đồ tại: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu biểu đồ: {str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ExcelChartTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
