from GUI_Ficron import * # import PyQt5 widgets
from PyQt5.QtWidgets import QFileDialog
import PyQt5
import count_util as c_util
import os
dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

#%%

def img_np2qt(img_np):
    height, width, channel = img_np.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(img_np, width, height, 3 * width, QtGui.QImage.Format_RGB888))


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.img_org_2ch = None
        self.img_org_2ch_rgb = None
        self.density_maps = None
        self.img_view = None
        self.path_dir = None


    def set_view_image(self, img_npRGB):
        img_view = img_np2qt(img_npRGB).scaled(800, 800, QtCore.Qt.KeepAspectRatio)
        self.view_image.setPixmap(img_view)

    def open_imag(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "Samples/", "Images (*.tif *.lif)",
                                                  options=options)
        self.img_org_2ch = c_util.read_tiff(fileName)[:, :, :2]
        img_tmp = c_util.convert_uint8(self.img_org_2ch)
        self.img_org_2ch_rgb = [c_util.convert_gray2bgr(img_tmp[:, :, i]) for i in range(2)]
        self.checkView_maps.setChecked(False)
        self.density_maps = None
        self.update_view()

    def result_count(self, data_result):
        self.res_par.setText(str(round(data_result[0], 1)))
        self.res_macrop.setText(str(round(data_result[1], 1)))
        self.res_inf.setText(str(round(data_result[2], 1)))


    def count_img(self):
        self.state_process.setText('Processing...')
        _, self.density_maps, result_cells = c_util.count_img(self.img_org_2ch, return_map=True, dcv=True)
        self.state_process.setText('Finished!')
        self.checkView_maps.setChecked(True)
        self.result_count(result_cells)
        self.update_view()

    def update_view(self):
        if self.img_org_2ch is None:
            return
        if self.checkCell.isChecked():
            channel = 0
        else:
            channel = 1
        if self.checkView_maps.isChecked() and self.density_maps is not None:
            self.img_view = c_util.merge_img_map(self.img_org_2ch_rgb[channel], self.density_maps)
        else:
            self.img_view = self.img_org_2ch_rgb[channel]

        self.set_view_image(self.img_view)

    def select_path_dir(self):
        options = QFileDialog.Options()
        path_dir = QFileDialog.getExistingDirectory(self, "Open Image", "Samples/", QFileDialog.ShowDirsOnly)
        self.path_dir = path_dir
        self.text_path_dir.setText('.../' + self.path_dir.split('/')[-1])

    def count_dir(self):
        pass
        

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()