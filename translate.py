from openai import OpenAI
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.pdf_parse_union_core_v2 import pdf_parse_union
from magic_pdf.config.enums import SupportedPdfParseMethod  # 导入枚举类
from loguru import logger

# 设置OpenAI API密钥
OpenAI.api_key = '1111'
client = OpenAI(
    base_url='http://101.126.68.179:20081/qwen2/v1',
    api_key=OpenAI.api_key)

def translate_text(text, source_lang, target_lang):
    """
    使用OpenAI API翻译文本
    """
    prompt = f"Translate the following {source_lang} text to {target_lang}: {text}"
    response = client.Completion.create(
        engine="Qwen2-72B",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()

def translate_pdf(pdf_path, output_path, source_lang, target_lang):
    """
    解析PDF文件，翻译文本，并将翻译后的文本填回到PDF文件中
    """
    # 读取PDF文件内容
    with open(pdf_path, 'rb') as file:
        pdf_bytes = file.read()

    # 创建PymuDocDataset对象
    dataset = PymuDocDataset(pdf_bytes)
    pdf_info_dict = pdf_parse_union(dataset, [], None, SupportedPdfParseMethod.TXT)

    # 遍历每一页
    for page in pdf_info_dict['pages']:
        # 翻译文本
        translated_text = translate_text(page['text'], source_lang, target_lang)

        # 将翻译后的文本填回到PDF文件中
        page['text'] = translated_text

    # 生成新的PDF文件
    with open(output_path, 'wb') as f:
        f.write(dataset.to_pdf(pdf_info_dict))

def init_model():
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        model_manager = ModelSingleton()
        txt_model = model_manager.get_model(False, False)
        logger.info(f"txt_model init final")
        ocr_model = model_manager.get_model(True, False)
        logger.info(f"ocr_model init final")
        return 0
    except Exception as e:
        logger.exception(e)
        return -1

model_init = init_model()
logger.info(f"model_init: {model_init}")

# 示例调用
translate_pdf('input.pdf', 'output.pdf', 'Chinese', 'English')