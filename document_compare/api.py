import frappe
import os
import json
from frappe.utils.file_manager import get_files_path
from werkzeug.utils import secure_filename
from . import utility
from . import qroq_compare_four
import frappe
import json
import re
from frappe import _
from frappe.utils import cint, flt
from frappe.model.document import Document
from frappe import db, throw, _, get_doc, get_list, get_cached_doc, ValidationError
from frappe.integrations.utils import make_get_request, make_post_request
from frappe.utils import get_datetime, now_datetime, get_link_to_form
from frappe import msgprint, log, get_traceback
from frappe.utils import get_datetime, now_datetime, get_link_to_form, cstr
from frappe.utils.html_utils import clean_html

@frappe.whitelist(allow_guest=True)
def upload_files():
    print(">>>>>>> upload_files")
    try:
        # Get uploaded files
        file1 = frappe.request.files.get('file1')

        file2 = frappe.request.files.get('file2')

        if not file1 or not file2:
            frappe.throw("Both files are required.")

        # Validate file extensions
        allowed_extensions = ['docx']
        if (file1.filename.split('.')[-1].lower() not in allowed_extensions or
                file2.filename.split('.')[-1].lower() not in allowed_extensions):
            frappe.throw("Only DOCX files are allowed.")

        # Secure filenames
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        # Get public file upload path
        upload_folder = get_files_path(is_private=0)
        file1_path = os.path.join(upload_folder, filename1)
        file2_path = os.path.join(upload_folder, filename2)

        # Save files
        with open(file1_path, 'wb') as f:
            f.write(file1.read())

        with open(file2_path, 'wb') as f:
            f.write(file2.read())

        # Process files
        html1 = utility.convert_docx_to_html(file1_path, upload_folder, True)
        html2 = utility.convert_docx_to_html(file2_path, upload_folder, False)


        html1 = utility.remove_empty_sections(html1)
        html2 = utility.remove_empty_sections(html2)

        html_data1 = utility.split_html_sections_in_list(
            utility.section_lines_with_section_using_bs4(html1))
        html_data2 = utility.split_html_sections_in_list(
            utility.section_lines_with_section_using_bs4(html2))

        

        html1 = "".join(utility.split_html_after_p_tags(html_data1))
        html2 = "".join(utility.split_html_after_p_tags(html_data2))

        return {
            "success": True,
            "message": "Files uploaded successfully",
            "file1": html1,
            "file2": html2,
            "filename1": filename1,
            "filename2": filename2
        }

    except Exception as e:
        frappe.log_error(f"File upload failed: {str(e)}")
        frappe.response.update({
            "success": False,
            "message": str(e)
        })

@frappe.whitelist(allow_guest=True)
def load_sections():
    if frappe.request.method != 'POST':
        # return JsonResponse({'message': 'Invalid request method'}, status=405)
        return {"message": _("Invalid request method"), "_server_messages": json.dumps([{"message": _("Invalid request method"), "indicator": "red"}])}, 405

    try:
        data = json.loads(frappe.request.data)
    except json.JSONDecodeError:
        # return JsonResponse({'message': 'Invalid JSON'}, status=400)
        return {"message": _("Invalid JSON"), "_server_messages": json.dumps([{"message": _("Invalid JSON"), "indicator": "red"}])}, 400

    # Validate required keys in JSON
    if 'filename1' not in data or 'filename2' not in data:
        # return JsonResponse({'message': 'Both files are required'}, status=400)
        return {"message": _("Both files are required"), "_server_messages": json.dumps([{"message": _("Both files are required"), "indicator": "red"}])}, 400

    if 'html1' not in data or 'html2' not in data:
        # return JsonResponse({'message': 'Both files are required'}, status=400)
        return {"message": _("Both files are required"), "_server_messages": json.dumps([{"message": _("Both files are required"), "indicator": "red"}])}, 400

    # document_id = data.get('document_id')
    # document_id = data.get('document_id')

    # # Construct file paths using the configured upload folder (or MEDIA_ROOT)
    # upload_folder = getattr(settings, 'UPLOAD_FOLDER', settings.MEDIA_ROOT)
    # file1_path = os.path.join(upload_folder, data['filename1'])
    # file2_path = os.path.join(upload_folder, data['filename2'])

    html1 = data['html1']
    html2 = data['html2']

    section_html1 = html1
    section_html2 = html2

    # Extract section data from the HTML using your utility functions
    section_data1 = utility.extract_sections_from_html(html1)
    section_data2 = utility.extract_sections_from_html(html2)

    # Compare the extracted sections using the groq_compare_four module
    section_details = qroq_compare_four.compare_with_groq(section_data1, section_data2, replace_threshold=0.600000)
    replaced_sections = [item for item in section_details if item['change_type'] == 'replaced']

    # Fix card_section bug before processing
    section_html1 = re.sub(r'class="\[\'card_section\'\]"', 'class="card_section"', section_html1)
    section_html2 = re.sub(r'class="\[\'card_section\'\]"', 'class="card_section"', section_html2)

    #####
    for section_item in replaced_sections:
        # if section_item['change_type'] == 'replaced':
        original_section_id = section_item['original_section_id']
        modified_section_id = section_item['modified_section_id']

        original_section = utility.find_section_and_section_style_in_section_html(section_html1, original_section_id)
        modified_section = utility.find_section_and_section_style_in_section_html(section_html2, modified_section_id)

        original_section_style = original_section['section_style'].replace(original_section_id, f"{original_section_id}" + "_{}")
        modified_section_style = modified_section['section_style'].replace(modified_section_id, f"{modified_section_id}" + "_{}")

        table_content = utility.is_table_content(original_section["updated_html"], modified_section["updated_html"])

        original_lines = utility.section_lines_using_bs4(original_section["updated_html"])
        modified_lines = utility.section_lines_using_bs4(modified_section["updated_html"])

        same_length = utility.is_same_length_of_lines(original_lines, modified_lines)

        # from main5 import analyze_section_changes, group_by_sections, generate_html_from_grouped_sections
        section_data = utility.analyze_section_changes(original_lines, modified_lines, same_length=same_length, table_content=table_content)

        grouped_sections = utility.group_by_sections(section_data, original_section_style, modified_section_style)

        generated_html = utility.generate_html_from_grouped_sections(grouped_sections)

        if '<tr>' in generated_html['original_html']:
            generated_html['original_html'] = utility.handle_table_html_using_bs4(generated_html['original_html'])

        if '<tr>' in generated_html['modified_html']:
            generated_html['modified_html'] = utility.handle_table_html_using_bs4(generated_html['modified_html'])

        generated_html['original_html'] = utility.remove_empty_sections(generated_html['original_html'])
        generated_html['modified_html'] = utility.remove_empty_sections(generated_html['modified_html'])

        section_html1 = utility.find_and_replace_section_element(section_html1, generated_html['original_html'], original_section_id)
        section_html2 = utility.find_and_replace_section_element(section_html2, generated_html['modified_html'], modified_section_id)

        section_data1 = utility.extract_sections_from_html(section_html1)
        section_data2 = utility.extract_sections_from_html(section_html2)

        # from groq_compare_four import compare_with_groq
        section_details = qroq_compare_four.compare_with_groq(section_data1, section_data2)

    ### add new code here----

    # response = {'success': True, 'message': 'Difference Found',
    #         'html1': section_html1, 'html2': section_html2, 'section_details': section_details}

    # Count Matching, Added, and Removed
    matched_count = sum(1 for item in section_details if item["change_type"] == "matched")
    added_count = sum(1 for item in section_details if item["change_type"] == "added")
    removed_count = sum(1 for item in section_details if item["change_type"] == "removed")
    replaced_count = sum(1 for item in section_details if item["change_type"] == "replaced")

    # Try to find an existing document
    document = frappe.get_value("Document", {"original_doc": data['filename1'], "modified_doc": data['filename2']}, "name")
    print(">>>>>>> document",document)

    if document:
        # Update existing document record
        # document = Document.objects.get(id=document_id)
        # document.original_doc = data['filename1']
        # document.modified_doc = data['filename2']
        # document.save()

        # Update existing document
        doc = frappe.get_doc("Document", document)
        doc.original_doc = data['filename1']
        doc.modified_doc = data['filename2']
        doc.save(ignore_permissions=True)

        # Update existing document comparison record
        # doc_compare = DocumentComparison.objects.get(document=document)
        doc_compare = get_doc("DocumentComparison", {"document": document})
        doc_compare.matched_count = matched_count
        doc_compare.added_count = added_count
        doc_compare.removed_count = removed_count
        doc_compare.replaced_count = replaced_count
        doc_compare.comparison_data = json.dumps({
            "section_details": section_details,
            "html1": section_html1,  # Store original HTML
            "html2": section_html2  # Store modified HTML
        })
        doc_compare.save(ignore_permissions=True)

        response = {
            'success': True,
            'message': _('Files updated successfully'),
            # 'user': document.user,
            # 'original_doc_name': document.original_doc,
            # 'modified_doc_name': document.modified_doc,
            # 'document_id': document.name,
            'file1': section_html1,
            'file2': section_html2,
            'section_details': section_details,
            'matched_count': matched_count,
            'added_count': added_count,
            'removed_count': removed_count,
            'replaced_count': replaced_count,
        }
    else:
         # Create new Document
        doc = frappe.get_doc({
            "doctype": "Document",
            "original_doc": data['filename1'],
            "modified_doc": data['filename2']
        })
        print(">>>>> doc", doc)
        doc.insert(ignore_permissions=True)

        # Create DocumentComparison
        doc_compare = frappe.get_doc({
            "doctype": "DocumentComparison",
            "document": doc.name,
            "matched_count": matched_count,
            "added_count": added_count,
            "removed_count": removed_count,
            "replaced_count": replaced_count,
            "comparison_data": json.dumps({
                "section_details": section_details,
                "html1": section_html1,
                "html2": section_html2
            })
        })
        doc_compare.insert(ignore_permissions=True)

         # Create Feedback Details
        feedback_details = frappe.get_doc({
            "doctype": "FeedbackDetails",
            "document": doc.name,
            "section_info": json.dumps(section_details)
        })
        feedback_details.insert(ignore_permissions=True)


        response = {
            'success': True,
            'message':  _('Files uploaded successfully'),
            # 'user': document.user,
            # 'original_doc_name': document.original_doc,
            # 'modified_doc_name': document.modified_doc,
            # 'document_id': document.name,
            'file1': section_html1,
            'file2': section_html2,
            'section_details': section_details,
            'matched_count': matched_count,
            'added_count': added_count,
            'removed_count': removed_count,
            'replaced_count': replaced_count,
        }
    return response
    # return {
    #     'success': True,
    #     'message':  _('Files uploaded successfully'),
    #     # 'user': document.user,
    #     # 'original_doc_name': original_doc,
    #     # 'modified_doc_name': modified_doc,
    #     # 'document_id': document.name,
    #     'file1': section_html1,
    #     'file2': section_html2,
    #     'section_details': section_details,
    #     'matched_count': matched_count,
    #     'added_count': added_count,
    #     'removed_count': removed_count,
    #     'replaced_count': replaced_count,
    # }

@frappe.whitelist(allow_guest=True)
def mark_sections():
    if frappe.request.method != 'POST':
        # return JsonResponse({'message': 'Invalid request method'}, status=405)
        return {"message": _("Invalid request method"), "_server_messages": json.dumps([{"message": _("Invalid request method"), "indicator": "red"}])}, 405
    
    try:
        data = json.loads(frappe.request.data)
    except json.JSONDecodeError:
        # return JsonResponse({'message': 'Invalid JSON'}, status=400)
        return {"message": _("Invalid JSON"), "_server_messages": json.dumps([{"message": _("Invalid JSON"), "indicator": "red"}])}, 400

    # Validate required keys in JSON
    if 'document_id' not in data or 'section_id' not in data or 'action' not in data:
        return {"message": _("Fields are required"), "_server_messages": json.dumps([{"message": _("Fields are required"), "indicator": "red"}])}, 400

    document_id = data['document_id']
    section_id = data['section_id']
    action = data['action']

    try:
        feedback_details = get_doc("Feedback Details", {"document": document_id})
    except frappe.DoesNotExistError:
        return {"message": _("Feedback Details not found"), "_server_messages": json.dumps([{"message": _("Feedback Details not found"), "indicator": "red"}])}, 404

    if feedback_details:
        if not feedback_details.section_info:
          feedback_details.section_info = {}

        feedback_details.section_info[section_id] = action
        feedback_details.save()

    response = {
        'success': True,
        'message': _('Section marked successfully'),
        'section_info': feedback_details.section_info
    }

    return response


def view_document(request):
    if request.method != 'POST':
        return {"message": _("Invalid request method"), "_server_messages": json.dumps([{"message": _("Invalid request method"), "indicator": "red"}])}, 405

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return {"message": _("Invalid JSON"), "_server_messages": json.dumps([{"message": _("Invalid JSON"), "indicator": "red"}])}, 400
    # Validate required keys in JSON
    if 'document_id' not in data:
        return {"message": _("Field is required"), "_server_messages": json.dumps([{"message": _("Field is required"), "indicator": "red"}])}, 400

    document_id = data['document_id']

    try:
        document = get_doc("Document", document_id)
    except frappe.DoesNotExistError:
        return {"message": _("Document not found"), "_server_messages": json.dumps([{"message": _("Document not found"), "indicator": "red"}])}, 404
    user = document.user
    original_doc = document.original_doc
    modified_doc = document.modified_doc

    try:
        doc_compare = get_doc("Document Comparison", {"document": document.name})
    except frappe.DoesNotExistError:
        return {"message": _("Document Comparison not found"), "_server_messages": json.dumps([{"message": _("Document Comparison not found"), "indicator": "red"}])}, 404

    matched_count = doc_compare.matched_count
    added_count = doc_compare.added_count
    removed_count = doc_compare.removed_count
    replaced_count = doc_compare.replaced_count
    comparison_data = doc_compare.comparison_data
    section_details = []
    original_html = ""
    modified_html = ""
    if comparison_data:
        section_details = comparison_data.get('section_details')
        original_html = comparison_data.get('html1')
        modified_html = comparison_data.get('html2')

    try:
        feedback_details = get_doc("Feedback Details", {"document": document.name})
        feedback_section_info = feedback_details.section_info
    except frappe.DoesNotExistError:
        feedback_section_info = {} #if feedback_details doc does not exist, set section_info to empty dict.

    response = {
        'success': True,
        'message': _('Compared data loads successfully'),
        'user': user,
        'original_doc_name': original_doc,
        'modified_doc_name': modified_doc,
        'document_id': document.name,
        'file1': original_html,
        'file2': modified_html,
        'section_details': section_details,
        'matched_count': matched_count,
        'added_count': added_count,
        'removed_count': removed_count,
        'replaced_count': replaced_count,
        'section_info': feedback_section_info
    }

    return response


def reanalyze_sections1(request):
    if request.method != 'POST':
        return {"message": _("Invalid request method"), "_server_messages": json.dumps([{"message": _("Invalid request method"), "indicator": "red"}])}, 405

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return {"message": _("Invalid JSON"), "_server_messages": json.dumps([{"message": _("Invalid JSON"), "indicator": "red"}])}, 400

    # Validate required keys in JSON
    if 'filename1' not in data or 'filename2' not in data:
        return {"message": _("Both files are required"), "_server_messages": json.dumps([{"message": _("Both files are required"), "indicator": "red"}])}, 400

    if 'html1' not in data or 'html2' not in data:
        return {"message": _("Both files are required"), "_server_messages": json.dumps([{"message": _("Both files are required"), "indicator": "red"}])}, 400

    if 'section_details' not in data:
        return {"message": _("Section details are required"), "_server_messages": json.dumps([{"message": _("Section details are required"), "indicator": "red"}])}, 400

    document_id = data.get('document_id')

    # # Construct file paths using the configured upload folder (or MEDIA_ROOT)
    # upload_folder = getattr(settings, 'UPLOAD_FOLDER', settings.MEDIA_ROOT)
    # file1_path = os.path.join(upload_folder, data['filename1'])
    # file2_path = os.path.join(upload_folder, data['filename2'])

    html1 = data['html1']
    html2 = data['html2']

    section_html1 = html1
    section_html2 = html2

    section_details = data['section_details']

    # # Extract section data from the HTML using your utility functions
    # section_data1 = utility.extract_sections_from_html(html1)
    # section_data2 = utility.extract_sections_from_html(html2)
    #
    # # Compare the extracted sections using the groq_compare_four module
    # section_details = compare_with_groq(section_data1, section_data2, replace_threshold=0.600000)
    removed_sections = [item for item in section_details if item['change_type'] == 'removed']
    added_sections = [item for item in section_details if item['change_type'] == 'added']
    matched_sections = [item for item in section_details if item['change_type'] == 'matched']
    # replaced_sections = [item for item in section_details if item['change_type'] == 'replaced']
    # print("removed_sections-->", removed_sections)
    # print("added_sections-->", added_sections)
    total_iterations = len(removed_sections) * len(added_sections)
    current_iteration = 0
    """
    for section_item in replaced_sections:
        # if section_item['change_type'] == 'replaced':
        original_section_id = section_item['original_section_id']
        modified_section_id = section_item['modified_section_id']

        original_section = utility.find_section_and_section_style_in_section_html(section_html1, original_section_id)
        modified_section = utility.find_section_and_section_style_in_section_html(section_html2, modified_section_id)

        # original_section_style = original_section['section_style'].replace(original_section_id, f"{original_section_id}" + "_{}")
        # modified_section_style = modified_section['section_style'].replace(modified_section_id, f"{modified_section_id}" + "_{}")

        table_content = utility.is_table_content(original_section["updated_html"], modified_section["updated_html"])

        original_lines = utility.section_lines_using_bs4(original_section["updated_html"])
        modified_lines = utility.section_lines_using_bs4(modified_section["updated_html"])

        same_length = utility.is_same_length_of_lines(original_lines, modified_lines)

        print("original_lines------->", original_lines)
        print("modified_lines------->", modified_lines)

        asdfgb
        # from main5 import analyze_section_changes, group_by_sections, generate_html_from_grouped_sections
        section_data = utility.analyze_section_changes(original_lines, modified_lines, same_length=same_length, table_content=table_content)

        grouped_sections = utility.group_by_sections(section_data, original_section_style, modified_section_style)

        generated_html = utility.generate_html_from_grouped_sections(grouped_sections)

        if '<tr>' in generated_html['original_html']:
            generated_html['original_html'] = utility.handle_table_html_using_bs4(generated_html['original_html'])

        if '<tr>' in generated_html['modified_html']:
            generated_html['modified_html'] = utility.handle_table_html_using_bs4(generated_html['modified_html'])

        generated_html['original_html'] = utility.remove_empty_sections(generated_html['original_html'])
        generated_html['modified_html'] = utility.remove_empty_sections(generated_html['modified_html'])

        section_html1 = utility.find_and_replace_section_element(section_html1, generated_html['original_html'], original_section_id)
        section_html2 = utility.find_and_replace_section_element(section_html2, generated_html['modified_html'], modified_section_id)

        section_data1 = utility.extract_sections_from_html(section_html1)
        section_data2 = utility.extract_sections_from_html(section_html2)

        # from groq_compare_four import compare_with_groq
        section_details = compare_with_groq(section_data1, section_data2)
    """

    section_matched = []
    for removed_item in removed_sections:
        for added_item in added_sections:
            current_iteration += 1
            print(f"Iteration {current_iteration}/{total_iterations}: Comparing {removed_item['original_section_id']} with {added_item['modified_section_id']}")
            # Extract the original and modified section IDs
            original_section_id = removed_item['original_section_id']
            modified_section_id = added_item['modified_section_id']

            # Find the original and modified sections in their respective HTML documents
            original_section = utility.find_section_and_section_style_in_section_html(html1, original_section_id)
            modified_section = utility.find_section_and_section_style_in_section_html(html2, modified_section_id)

            if original_section is None or modified_section is None:
                continue

            # original_section_lines = utility.section_lines_using_bs4(original_section["updated_html"])
            # modified_section_lines = utility.section_lines_using_bs4(modified_section["updated_html"])
            # # if "org_section_26_1" in original_section_id and "mod_section_27_1" in modified_section_id:
            # print("original_section_updated_html----->", original_section_lines)
            # print("modified_section_updated_html----->", modified_section_lines)
            section_result = qroq_compare_four.check_similarity(original_section["updated_html"], modified_section["updated_html"])

            if section_result['is_match']:
                section = {"change_type": "matched", "original_section_id": original_section_id, "modified_section_id": modified_section_id, "score": section_result['score']}
                section_matched.append(section)

    # print("matched_section------>", matched_section)
    print("section_matched------>", section_matched)

    duplicate_matches = utility.find_duplicate_matches_in_matched_sections(section_matched)

    grouped_matches = utility.group_duplicate_matches_in_matched_sections(duplicate_matches)

    scores_by_group = utility.extract_scores_from_groups_in_matched_sections(grouped_matches)

    best_score_list = []
    if scores_by_group:
        for group_id, scores in scores_by_group.items():
            best_score_list.append(utility.determine_best_matched_section(scores))
    print("best_score_list------>", best_score_list)
    print("section_matched------>", section_matched)
    # best_scores_set = {item['score'] for item in best_score_list}
    filtered_matched_section = [item for item in section_matched if item['score'] not in best_score_list]
    filtered_matched_section = utility.remove_duplicate_sections_ids_from_filtered_matched_section(filtered_matched_section)
    updated_matched_section = filtered_matched_section + matched_sections
    # print("removed_sections------>", removed_sections)
    # print("added_sections------>", added_sections)
    print("filtered_matched_section------>", filtered_matched_section)

    # Extract original_section_ids from filtered_matched_section
    original_matched_ids = {item["original_section_id"] for item in updated_matched_section}
    modified_matched_ids = {item["modified_section_id"] for item in updated_matched_section}

    removed_original_ids = {item['original_section_id'] for item in section_details if item['original_section_id'] not in original_matched_ids}
    added_modified_ids = {item['modified_section_id'] for item in section_details if item['modified_section_id'] not in modified_matched_ids}

    # {'change_type': 'matched', 'original_section_id': 'org_section_1_1_1', 'modified_section_id': 'mod_section_1_1_1'},
    updated_removed_section = []
    for id in removed_original_ids:
        item = {'change_type': 'removed', 'original_section_id': id, 'modified_section_id': None}
        updated_removed_section.append(item)

    updated_added_section = []
    for id in added_modified_ids:
        item = {'change_type': 'added', 'original_section_id': None, 'modified_section_id': id}
        updated_added_section.append(item)

    # removed_sections = [item for item in removed_sections if item["original_section_id"] not in original_matched_ids]
    # added_sections = [item for item in added_sections if item["modified_section_id"] not in modified_matched_ids]

    # for item in section_details[:]:
    #     if item["original_section_id"] in original_matched_ids:
    #         section_details.remove(item)  # Remove matching item
    #
    # for item in section_details[:]:
    #     if item["modified_section_id"] in modified_matched_ids:
    #         section_details.remove(item)  # Remove matching item

    merge_section_details = updated_matched_section + updated_removed_section + updated_added_section
    # combined_section_details = utility.merge_section_details(section_details, matched_sections_data)

    # matched_sections =
    # section_details = matched_sections + removed_sections + added_sections

    # print("merge_section_details------>", merge_section_details)

    """
    for matched_item in filtered_matched_section:
        # Remove the item with original_section_id and change_type 'removed'
        section_details = [
            item for item in section_details
            if not (item['original_section_id'] == matched_item['original_section_id'] and item['change_type'] == 'removed')
        ]

        # Remove the item with modified_section_id and change_type 'added'
        section_details = [
            item for item in section_details
            if not (item['modified_section_id'] == matched_item['modified_section_id'] and item['change_type'] == 'added')
        ]

        # Step 2: Add the item from matched_section to section_details
        section_details.append(matched_item)
"""
    """matched_modified_ids = [item["modified_section_id"] for item in matched_section if item["change_type"] == "matched"]

    filtered_matched_section = [
        item for item in matched_section
        if not (item["change_type"] == "added" and item["modified_section_id"] in matched_modified_ids)
    ]

    # Update section_details based on matched_section
    for match in filtered_matched_section:
        for section in section_details:
            if section["original_section_id"] == match["original_section_id"]:
                section["change_type"] = match["change_type"]
                section["modified_section_id"] = match["modified_section_id"]

    print(section_details)
"""
    # response = {'success': True, 'message': 'Difference Found',
    #         'html1': section_html1, 'html2': section_html2, 'section_details': section_details}

    # Count Matching, Added, and Removed
    matched_count = sum(1 for item in merge_section_details if item["change_type"] == "matched")
    added_count = sum(1 for item in merge_section_details if item["change_type"] == "added")
    removed_count = sum(1 for item in merge_section_details if item["change_type"] == "removed")
    replaced_count = sum(1 for item in merge_section_details if item["change_type"] == "replaced")
    """
    if document_id:
        # Update existing document record
        document = Document.objects.get(id=document_id)
        document.original_doc = data['filename1']
        document.modified_doc = data['filename2']
        document.save()

        # Update existing document comparison record
        doc_compare = DocumentComparison.objects.get(document=document)
        doc_compare.matched_count = matched_count
        doc_compare.added_count = added_count
        doc_compare.removed_count = removed_count
        doc_compare.replaced_count = replaced_count
        doc_compare.comparison_data = {
            "section_details": section_details,
            "html1": section_html1,  # Store original HTML
            "html2": section_html2  # Store modified HTML
        }
        doc_compare.save()

        # Update feedback details (assuming additional fields exist)
        feedback_details = FeedbackDetails.objects.get(document=document)
        feedback_details.save()  # If other fields need updates, modify before saving.

        response = {
            'success': True,
            'message': 'Files updated successfully',
            'user': 1, #document.user.id,
            'original_doc_name': data['filename1'], #document.original_doc.name,
            'modified_doc_name': data['filename2'], #document.modified_doc.name,
            'document_id': document_id, # document.id,
            'file1': section_html1,
            'file2': section_html2,
            'section_details': section_details,
            'matched_count': matched_count,
            'added_count': added_count,
            'removed_count': removed_count,
            'replaced_count': replaced_count,
        }

    else:
        # Create new records as per your original code
        document = Document.objects.create(
            user=User.objects.get(username='admin'),
            original_doc=data['filename1'],
            modified_doc=data['filename2']
        )

        doc_compare = DocumentComparison.objects.create(
            document=document,
            matched_count=matched_count,
            added_count=added_count,
            removed_count=removed_count,
            replaced_count=replaced_count,
            comparison_data={
                "section_details": section_details,
                "html1": section_html1,
                "html2": section_html2
            }
        )

        feedback_details = FeedbackDetails.objects.create(document=document)

        response = {
            'success': True,
            'message': 'Files uploaded successfully',
            'user': document.user.id,
            'original_doc_name': document.original_doc.name,
            'modified_doc_name': document.modified_doc.name,
            'document_id': document.id,
            'file1': section_html1,
            'file2': section_html2,
            'section_details': section_details,
            'matched_count': matched_count,
            'added_count': added_count,
            'removed_count': removed_count,
            'replaced_count': replaced_count,
        }
    """
    response = {
        'success': True,
        'message': _('Files updated successfully'),
        'user': frappe.session.user,
        'original_doc_name': data['filename1'],
        'modified_doc_name': data['filename2'],
        'document_id': document_id,
        'file1': section_html1,
        'file2': section_html2,
        'section_details': merge_section_details,
        'matched_count': matched_count,
        'added_count': added_count,
        'removed_count': removed_count,
        'replaced_count': replaced_count,
    }
    return response

def reanalyze_sections(request):
    if request.method != 'POST':
        return JsonResponse({'message': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'message': 'Invalid JSON'}, status=400)

    # Validate required keys in JSON
    if 'filename1' not in data or 'filename2' not in data:
        return JsonResponse({'message': 'Both files are required'}, status=400)

    if 'html1' not in data or 'html2' not in data:
        return JsonResponse({'message': 'Both files are required'}, status=400)

    if 'section_details' not in data:
        return JsonResponse({'message': 'Section details are required'}, status=400)

    document_id = data.get('document_id')

    # # Construct file paths using the configured upload folder (or MEDIA_ROOT)
    # upload_folder = getattr(settings, 'UPLOAD_FOLDER', settings.MEDIA_ROOT)
    # file1_path = os.path.join(upload_folder, data['filename1'])
    # file2_path = os.path.join(upload_folder, data['filename2'])

    html1 = data['html1']
    html2 = data['html2']

    section_html1 = html1
    section_html2 = html2

    section_details = data['section_details']

    # # Extract section data from the HTML using your utility functions
    # section_data1 = utility.extract_sections_from_html(html1)
    # section_data2 = utility.extract_sections_from_html(html2)
    #
    # # Compare the extracted sections using the groq_compare_four module
    # section_details = compare_with_groq(section_data1, section_data2, replace_threshold=0.600000)
    removed_sections = [item for item in section_details if item['change_type'] == 'removed']
    added_sections = [item for item in section_details if item['change_type'] == 'added']
    matched_sections = [item for item in section_details if item['change_type'] == 'matched']
    # replaced_sections = [item for item in section_details if item['change_type'] == 'replaced']
    # print("removed_sections-->", removed_sections)
    # print("added_sections-->", added_sections)
    total_iterations = len(removed_sections) * len(added_sections)
    current_iteration = 0
    """
    for section_item in replaced_sections:
        # if section_item['change_type'] == 'replaced':
        original_section_id = section_item['original_section_id']
        modified_section_id = section_item['modified_section_id']

        original_section = utility.find_section_and_section_style_in_section_html(section_html1, original_section_id)
        modified_section = utility.find_section_and_section_style_in_section_html(section_html2, modified_section_id)

        # original_section_style = original_section['section_style'].replace(original_section_id, f"{original_section_id}" + "_{}")
        # modified_section_style = modified_section['section_style'].replace(modified_section_id, f"{modified_section_id}" + "_{}")

        table_content = utility.is_table_content(original_section["updated_html"], modified_section["updated_html"])

        original_lines = utility.section_lines_using_bs4(original_section["updated_html"])
        modified_lines = utility.section_lines_using_bs4(modified_section["updated_html"])

        same_length = utility.is_same_length_of_lines(original_lines, modified_lines)

        print("original_lines------->", original_lines)
        print("modified_lines------->", modified_lines)

        asdfgb
        # from main5 import analyze_section_changes, group_by_sections, generate_html_from_grouped_sections
        section_data = utility.analyze_section_changes(original_lines, modified_lines, same_length=same_length, table_content=table_content)

        grouped_sections = utility.group_by_sections(section_data, original_section_style, modified_section_style)

        generated_html = utility.generate_html_from_grouped_sections(grouped_sections)

        if '<tr>' in generated_html['original_html']:
            generated_html['original_html'] = utility.handle_table_html_using_bs4(generated_html['original_html'])

        if '<tr>' in generated_html['modified_html']:
            generated_html['modified_html'] = utility.handle_table_html_using_bs4(generated_html['modified_html'])

        generated_html['original_html'] = utility.remove_empty_sections(generated_html['original_html'])
        generated_html['modified_html'] = utility.remove_empty_sections(generated_html['modified_html'])

        section_html1 = utility.find_and_replace_section_element(section_html1, generated_html['original_html'], original_section_id)
        section_html2 = utility.find_and_replace_section_element(section_html2, generated_html['modified_html'], modified_section_id)

        section_data1 = utility.extract_sections_from_html(section_html1)
        section_data2 = utility.extract_sections_from_html(section_html2)

        # from groq_compare_four import compare_with_groq
        section_details = compare_with_groq(section_data1, section_data2)
    """

    section_matched = []
    modified_matched = []
    for removed_item in removed_sections:
        original_matched = []
        for added_item in added_sections:
            current_iteration += 1
            print(f"Iteration {current_iteration}/{total_iterations}: Comparing {removed_item['original_section_id']} with {added_item['modified_section_id']}")
            # Extract the original and modified section IDs
            original_section_id = removed_item['original_section_id']
            modified_section_id = added_item['modified_section_id']

            # Find the original and modified sections in their respective HTML documents
            original_section = utility.find_section_and_section_style_in_section_html(html1, original_section_id)
            modified_section = utility.find_section_and_section_style_in_section_html(html2, modified_section_id)

            if original_section is None or modified_section is None:
                continue

            # original_section_lines = utility.section_lines_using_bs4(original_section["updated_html"])
            # modified_section_lines = utility.section_lines_using_bs4(modified_section["updated_html"])
            # # if "org_section_26_1" in original_section_id and "mod_section_27_1" in modified_section_id:
            # print("original_section_updated_html----->", original_section_lines)
            # print("modified_section_updated_html----->", modified_section_lines)
            section_result = qroq_compare_four.check_similarity(original_section["updated_html"], modified_section["updated_html"])

            if section_result['is_match']:
                section = {"change_type": "matched", "original_section_id": original_section_id, "modified_section_id": modified_section_id, "score": section_result['score']}
                # section_matched.append(section)
                original_matched.append(section)

        score_list = [item['score'] for item in original_matched]
        if score_list:
            best_score = utility.determine_best_matched_section(score_list)
            best_item = [item for item in original_matched if item['score'] == best_score]
            modified_matched.append(best_item[0])
            print("best_item------>", best_item)
    # modified_matched = [{"change_type":"matched","original_section_id":"org_section_26_1_0","modified_section_id":"mod_section_230_1_p_1_0","score":{"entailment":0.9459333419799805,"contradiction":0.007595316506922245,"length":91.42857142857143}},{"change_type":"matched","original_section_id":"org_section_26_2_0","modified_section_id":"mod_section_2_2_p_1_0","score":{"entailment":0.8183491230010986,"contradiction":0.0699523314833641,"length":89.66836734693877}},{"change_type":"matched","original_section_id":"org_section_26_3_0","modified_section_id":"mod_section_2_2_p_1_0","score":{"entailment":0.8626417517662048,"contradiction":0.03757418692111969,"length":82.9302987197724}},{"change_type":"matched","original_section_id":"org_section_26_4_li_2","modified_section_id":"mod_section_2_2_p_1_0","score":{"entailment":0.6533458828926086,"contradiction":0.06356573849916458,"length":91.32290184921764}},{"change_type":"matched","original_section_id":"org_section_76_0","modified_section_id":"mod_section_223_0","score":{"entailment":0.9718410968780518,"contradiction":0.00346723897382617,"length":98.82352941176471}},{"change_type":"matched","original_section_id":"org_section_76_2","modified_section_id":"mod_section_223_2","score":{"entailment":0.43190664052963257,"contradiction":0.2610868811607361,"length":71.69811320754718}},{"change_type":"matched","original_section_id":"org_section_80_1_p_1_0","modified_section_id":"mod_section_230_1_p_1_0","score":{"entailment":0.9299633502960205,"contradiction":0.010944325476884842,"length":80.0}},{"change_type":"matched","original_section_id":"org_section_80_1_p_2","modified_section_id":"mod_section_230_1_p_1_0","score":{"entailment":0.6116349101066589,"contradiction":0.23176436126232147,"length":97.22222222222221}},{"change_type":"matched","original_section_id":"org_section_96_0","modified_section_id":"mod_section_78_1","score":{"entailment":0.746116042137146,"contradiction":0.04302623122930527,"length":33.18903318903319}},{"change_type":"matched","original_section_id":"org_section_105_1","modified_section_id":"mod_section_279_1_0","score":{"entailment":0.5680039525032043,"contradiction":0.03677826747298241,"length":57.14285714285714}},{"change_type":"matched","original_section_id":"org_section_108_0","modified_section_id":"mod_section_257","score":{"entailment":0.34916970133781433,"contradiction":0.3047263026237488,"length":78.343949044586}}]
    # print("modified_matched------>", modified_matched)

    from collections import defaultdict
    modified_section_data = defaultdict(list)
    for item in modified_matched:
        modified_section_data[item["modified_section_id"]].append(item)
    modified_section_data = dict(modified_section_data)

    for keys, values in modified_section_data.items():
        if len(values) > 1:
            score_list = [item['score'] for item in values]
            print("score_list--", score_list)
            if score_list:
                best_score = utility.determine_best_matched_section(score_list)
                best_item = [item for item in modified_matched if item['score'] == best_score]
                section_matched.append(best_item[0])
        else:
            # print("values--", values)
            section_matched.append(values[0])
    print("section_matched------>", section_matched)

    # modified_matched = []
    # for item in section_matched:
    #     modified_matched.append(item['modified_section_id'])
    #     if item['modified_section_id'] in modified_matched:
    #         pass

    # duplicate_matches = utility.find_duplicate_matches_in_matched_sections(section_matched)
    #
    # grouped_matches = utility.group_duplicate_matches_in_matched_sections(duplicate_matches)
    #
    # scores_by_group = utility.extract_scores_from_groups_in_matched_sections(grouped_matches)
    #
    # best_score_list = []
    # if scores_by_group:
    #     for group_id, scores in scores_by_group.items():
    #         best_score_list.append(utility.determine_best_matched_section(scores))
    # print("best_score_list------>", best_score_list)
    # print("section_matched------>", section_matched)
    # best_scores_set = {item['score'] for item in best_score_list}
    # filtered_matched_section = [item for item in section_matched if item['score'] not in best_score_list]
    # filtered_matched_section = utility.remove_duplicate_sections_ids_from_filtered_matched_section(filtered_matched_section)
    updated_matched_section = section_matched + matched_sections
    # print("removed_sections------>", removed_sections)
    # print("added_sections------>", added_sections)
    # print("updated_matched_section------>", updated_matched_section)

    # Extract original_section_ids from filtered_matched_section
    original_matched_ids = {item["original_section_id"] for item in updated_matched_section}
    modified_matched_ids = {item["modified_section_id"] for item in updated_matched_section}

    removed_original_ids = {item['original_section_id'] for item in removed_sections if item['original_section_id'] not in original_matched_ids}
    added_modified_ids = {item['modified_section_id'] for item in added_sections if item['modified_section_id'] not in modified_matched_ids}

    # {'change_type': 'matched', 'original_section_id': 'org_section_1_1_1', 'modified_section_id': 'mod_section_1_1_1'},
    updated_removed_section = []
    for id in removed_original_ids:
        item = {'change_type': 'removed', 'original_section_id': id, 'modified_section_id': None}
        updated_removed_section.append(item)

    updated_added_section = []
    for id in added_modified_ids:
        item = {'change_type': 'added', 'original_section_id': None, 'modified_section_id': id}
        updated_added_section.append(item)

    # removed_sections = [item for item in removed_sections if item["original_section_id"] not in original_matched_ids]
    # added_sections = [item for item in added_sections if item["modified_section_id"] not in modified_matched_ids]

    # for item in section_details[:]:
    #     if item["original_section_id"] in original_matched_ids:
    #         section_details.remove(item)  # Remove matching item
    #
    # for item in section_details[:]:
    #     if item["modified_section_id"] in modified_matched_ids:
    #         section_details.remove(item)  # Remove matching item

    merge_section_details = updated_matched_section + updated_removed_section + updated_added_section
    # combined_section_details = utility.merge_section_details(section_details, matched_sections_data)

    # matched_sections =
    # section_details = matched_sections + removed_sections + added_sections

    # print("merge_section_details------>", merge_section_details)

    """
    for matched_item in filtered_matched_section:
        # Remove the item with original_section_id and change_type 'removed'
        section_details = [
            item for item in section_details
            if not (item['original_section_id'] == matched_item['original_section_id'] and item['change_type'] == 'removed')
        ]

        # Remove the item with modified_section_id and change_type 'added'
        section_details = [
            item for item in section_details
            if not (item['modified_section_id'] == matched_item['modified_section_id'] and item['change_type'] == 'added')
        ]

        # Step 2: Add the item from matched_section to section_details
        section_details.append(matched_item)
"""
    """matched_modified_ids = [item["modified_section_id"] for item in matched_section if item["change_type"] == "matched"]

    filtered_matched_section = [
        item for item in matched_section
        if not (item["change_type"] == "added" and item["modified_section_id"] in matched_modified_ids)
    ]

    # Update section_details based on matched_section
    for match in filtered_matched_section:
        for section in section_details:
            if section["original_section_id"] == match["original_section_id"]:
                section["change_type"] = match["change_type"]
                section["modified_section_id"] = match["modified_section_id"]

    print(section_details)
"""
    # response = {'success': True, 'message': 'Difference Found',
    #         'html1': section_html1, 'html2': section_html2, 'section_details': section_details}

    # Count Matching, Added, and Removed
    matched_count = sum(1 for item in merge_section_details if item["change_type"] == "matched")
    added_count = sum(1 for item in merge_section_details if item["change_type"] == "added")
    removed_count = sum(1 for item in merge_section_details if item["change_type"] == "removed")
    replaced_count = sum(1 for item in merge_section_details if item["change_type"] == "replaced")

    if document_id:
        # Update existing document record
        document = Document.objects.get(id=document_id)
        document.original_doc = data['filename1']
        document.modified_doc = data['filename2']
        document.save()

        # Update existing document comparison record
        doc_compare = DocumentComparison.objects.get(document=document)
        doc_compare.matched_count = matched_count
        doc_compare.added_count = added_count
        doc_compare.removed_count = removed_count
        doc_compare.replaced_count = replaced_count
        doc_compare.comparison_data = {
            "section_details": merge_section_details,
            "html1": section_html1,  # Store original HTML
            "html2": section_html2  # Store modified HTML
        }
        doc_compare.save()

        # Update feedback details (assuming additional fields exist)
        feedback_details = FeedbackDetails.objects.get(document=document)
        feedback_details.save()  # If other fields need updates, modify before saving.

        response = {
            'success': True,
            'message': 'Files updated successfully',
            'user': document.user.id, #1,
            'original_doc_name': document.original_doc.name, #data['filename1'],
            'modified_doc_name': document.modified_doc.name, #data['filename2'],
            'document_id': document.id, # document_id,
            'file1': section_html1,
            'file2': section_html2,
            'section_details': merge_section_details,
            'matched_count': matched_count,
            'added_count': added_count,
            'removed_count': removed_count,
            'replaced_count': replaced_count,
        }

    else:
        # Create new records as per your original code
        document = Document.objects.create(
            user=User.objects.get(username='admin'),
            original_doc=data['filename1'],
            modified_doc=data['filename2']
        )

        doc_compare = DocumentComparison.objects.create(
            document=document,
            matched_count=matched_count,
            added_count=added_count,
            removed_count=removed_count,
            replaced_count=replaced_count,
            comparison_data={
                "section_details": section_details,
                "html1": section_html1,
                "html2": section_html2
            }
        )

        feedback_details = FeedbackDetails.objects.create(document=document)

        response = {
            'success': True,
            'message': 'Files uploaded successfully',
            'user': document.user.id,
            'original_doc_name': document.original_doc.name,
            'modified_doc_name': document.modified_doc.name,
            'document_id': document.id,
            'file1': section_html1,
            'file2': section_html2,
            'section_details': merge_section_details,
            'matched_count': matched_count,
            'added_count': added_count,
            'removed_count': removed_count,
            'replaced_count': replaced_count,
        }

    # response = {
    #     'success': True,
    #     'message': 'Files updated successfully',
    #     'user': 1,  # document.user.id,
    #     'original_doc_name': data['filename1'],  # document.original_doc.name,
    #     'modified_doc_name': data['filename2'],  # document.modified_doc.name,
    #     'document_id': document_id,  # document.id,
    #     'file1': section_html1,
    #     'file2': section_html2,
    #     'section_details': merge_section_details,
    #     'matched_count': matched_count,
    #     'added_count': added_count,
    #     'removed_count': removed_count,
    #     'replaced_count': replaced_count,
    # }
    return JsonResponse(response)
