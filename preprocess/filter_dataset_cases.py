"""
   2.                   id       
"""
import openpyxl

def filter_scapula_points_dataset(file_path):
    """
           scapula    hemerus            
    """
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active  # Internal processing step.
    
    # Internal processing step.
    for row in range(2, 107):
        b_value = ws[f'B{row}'].value
        c_value = ws[f'C{row}'].value
        l_value = ws[f'L{row}'].value
        
        # Internal processing step.
        if b_value != -1 and c_value == 0 and l_value == '18':
            ws[f'O{row}'] = 1
        else:
            ws[f'O{row}'] = 0
    
    # Internal processing step.
    wb.save(file_path)
    print("Filtering complete. The dataset has been updated.")


if __name__ == "__main__":
    file_path = "<DATA_ROOT>"
    
    filter_scapula_points_dataset(file_path)
