
def construct_page_component(page_id, image_info, table_info,  text_blocks_preproc, layout_bboxes, inline_eq_info, interline_eq_info, raw_pymu_blocks, 
                             removed_text_blocks, removed_image_blocks, images_backup, droped_table_block, table_backup,layout_tree,
                             page_w, page_h, footnote_bboxes_tmp):
    """
    
    """
    return_dict = {}
    
    return_dict['para_blocks'] = {}
    return_dict['preproc_blocks'] = text_blocks_preproc
    return_dict['images'] = image_info
    return_dict['tables'] = table_info
    return_dict['interline_equations'] = interline_eq_info
    return_dict['inline_equations'] = inline_eq_info
    return_dict['layout_bboxes'] = layout_bboxes
    return_dict['pymu_raw_blocks'] = raw_pymu_blocks
    return_dict['global_statistic'] = {}
    
    return_dict['droped_text_block'] = removed_text_blocks
    return_dict['droped_image_block'] = removed_image_blocks
    return_dict['droped_table_block'] = []
    return_dict['image_backup'] = images_backup
    return_dict['table_backup'] = []    
    return_dict['page_idx'] = page_id
    return_dict['page_size'] = [page_w, page_h]
    return_dict['_layout_tree'] = layout_tree # 辅助分析layout作用
    return_dict['footnote_bboxes_tmp'] = footnote_bboxes_tmp
    
    return return_dict
