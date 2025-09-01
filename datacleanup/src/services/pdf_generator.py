"""PDF report generator with cost savings and infrastructure recommendations."""
import io
import base64
from typing import Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_CENTER

class PDFReportGenerator:
    """Enhanced PDF report generator with cost savings and infrastructure guidance."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()

    def setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#2c3e50'),
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=HexColor('#34495e'),
        ))
        self.styles.add(ParagraphStyle(
            'HighlightBox',
            parent=self.styles['Normal'],
            fontSize=12,
            backColor=HexColor('#ecf0f1'),
            borderWidth=1,
            borderColor=HexColor('#bdc3c7'),
            borderPadding=10,
            spaceAfter=12
        ))

    def create_cost_savings_chart(self, costs_data: Dict[str, Any]) -> str:
        """Create cost comparison chart and return as base64 string."""
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['All-OpenAI', 'Hybrid AI']
        costs = [
            costs_data.get('cost_all_openai', 0),
            costs_data.get('cost_hybrid', 0)
        ]

        bars = ax.bar(categories, costs, alpha=0.8, width=0.6)

        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(costs) * 0.01,
                   f'${cost:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Add savings annotation
        savings = costs_data.get('savings_absolute', 0)
        savings_pct = costs_data.get('savings_pct', 0) * 100

        if savings > 0 and costs[0] > costs[1]:
            arrow = patches.FancyArrowPatch((0, costs[0]), (1, costs[1]),
                                          arrowstyle='<->', mutation_scale=20,
                                          linewidth=2)
            ax.add_patch(arrow)

            mid_y = (costs[0] + costs[1]) / 2
            ax.text(0.5, mid_y, f'Saves ${savings:.4f}\\n({savings_pct:.1f}%)',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_ylabel('Cost (USD)', fontsize=12)
        ax.set_title(f'Cost Comparison - {costs_data.get("model", "gpt-4o-mini")}', 
                    fontsize=14, fontweight='bold', pad=20)

        ax.set_ylim(0, max(costs) * 1.2 if max(costs) > 0 else 1)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return chart_b64

    def create_infrastructure_recommendations(self, concurrent_users: int = 1000) -> Dict[str, str]:
        """Generate infrastructure recommendations for specified concurrent users."""
        base_cpu_cores = max(8, concurrent_users // 125)  # 1 core per ~125 users
        base_ram_gb = max(16, int(concurrent_users // 62.5))   # 1GB per ~62.5 users
        base_storage_gb = max(500, int(concurrent_users * 0.5)) # 0.5GB per user for temp files

        recommendations = {
            "server_type": "Ubuntu 22.04 LTS or CentOS Stream 9",
            "cpu_cores": f"{base_cpu_cores} vCPU cores minimum",
            "ram": f"{int(base_ram_gb)} GB RAM minimum",
            "storage": f"{int(base_storage_gb)} GB SSD storage",
            "network": "1 Gbps network connection minimum",
        }

        if concurrent_users >= 500:
            recommendations["load_balancer"] = "Required - HAProxy or NGINX"
            recommendations["scaling"] = "Kubernetes recommended for auto-scaling"
            recommendations["replicas"] = f"{max(3, concurrent_users // 300)} app replicas minimum"
            recommendations["database"] = "PostgreSQL with read replicas"
            recommendations["caching"] = "Redis cluster for session management"
        else:
            recommendations["load_balancer"] = "Optional - single NGINX instance sufficient"
            recommendations["scaling"] = "Docker Compose adequate, Kubernetes optional"
            recommendations["replicas"] = "2-3 app replicas for availability"
            recommendations["database"] = "PostgreSQL single instance with backups"
            recommendations["caching"] = "Single Redis instance"

        if concurrent_users >= 2000:
            recommendations["cdn"] = "CDN required (CloudFlare, AWS CloudFront)"
            recommendations["monitoring"] = "Full observability stack (Prometheus, Grafana, ELK)"
            recommendations["backup"] = "Real-time backup with geographic distribution"

        return recommendations

    def generate_report(
        self,
        summary_data: Dict[str, Any],
        output_path: str,
        original_filename: str = "data.csv"
    ) -> str:
        """Generate comprehensive PDF report with cost savings and infrastructure guidance."""
        doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.5*inch)
        story = []

        title = Paragraph("Data Cleanup - Executive Summary", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))

        # Processing Overview
        story.append(Paragraph("Processing Overview", self.styles['SectionHeader']))

        overview_data = [
            ['Original File', original_filename],
            ['Processing Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Total Rows Processed', f"{summary_data.get('rows_processed', 0):,}"],
            ['Duplicates Removed', f"{summary_data.get('duplicates_removed', 0):,}"],
            ['Data Quality Score', f"{summary_data.get('data_quality_score', 0):.1f}%"]
        ]

        overview_table = Table(overview_data, colWidths=[2.5*inch, 3*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7'))
        ]))

        story.append(overview_table)
        story.append(Spacer(1, 30))

        # Cost Savings Analysis
        categorization = summary_data.get('categorization', {})
        if categorization.get('enabled') and 'costs' in categorization:
            story.append(Paragraph("💰 Cost Savings Analysis", self.styles['SectionHeader']))

            costs_data = categorization['costs']

            savings_summary = summary_data.get('savings_summary', '')
            if savings_summary:
                savings_para = Paragraph(f"<b>Summary:</b> {savings_summary}", self.styles['HighlightBox'])
                story.append(savings_para)

            cost_table_data = [
                ['Metric', 'Value'],
                ['AI Model Used', costs_data.get('model', 'N/A')],
                ['Input Tokens/Row', f"{costs_data.get('assumptions', {}).get('input_tokens_per_row', 0)}"],
                ['Output Tokens/Row', f"{costs_data.get('assumptions', {}).get('output_tokens_per_row', 0)}"],
                ['All-OpenAI Cost', f"${costs_data.get('cost_all_openai', 0):.4f}"],
                ['Hybrid AI Cost', f"${costs_data.get('cost_hybrid', 0):.4f}"],
                ['Absolute Savings', f"${costs_data.get('savings_absolute', 0):.4f}"],
                ['Percentage Savings', f"{costs_data.get('savings_pct', 0)*100:.1f}%"],
                ['HF Processed Rows', f"{costs_data.get('rows_breakdown', {}).get('hf', 0):,}"],
                ['OpenAI Fallback Rows', f"{costs_data.get('rows_breakdown', {}).get('openai', 0):,}"]
            ]

            cost_table = Table(cost_table_data, colWidths=[2.5*inch, 2*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2980b9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
                ('BACKGROUND', (0, 6), (-1, 7), HexColor('#d5e8d4')),
                ('FONTNAME', (0, 6), (-1, 7), 'Helvetica-Bold')
            ]))

            story.append(cost_table)
            story.append(Spacer(1, 20))

            try:
                chart_b64 = self.create_cost_savings_chart(costs_data)
                chart_image = Image(io.BytesIO(base64.b64decode(chart_b64)), width=6*inch, height=3.6*inch)
                story.append(chart_image)
                story.append(Spacer(1, 20))
            except Exception:
                pass

            footnote = Paragraph(
                "<i>*Hugging Face inference assumed $0 variable cost. Excludes fixed hosting and infrastructure costs.</i>",
                self.styles['Normal']
            )
            story.append(footnote)
            story.append(Spacer(1, 30))

        # Infrastructure Recommendations
        story.append(Paragraph("🏗️ Infrastructure Recommendations", self.styles['SectionHeader']))

        infra_recommendations = self.create_infrastructure_recommendations(1000)
        infra_table_data = [
            ['Component', 'Specification'],
            ['Server OS', infra_recommendations['server_type']],
            ['CPU Requirements', infra_recommendations['cpu_cores']],
            ['Memory (RAM)', infra_recommendations['ram']],
            ['Storage', infra_recommendations['storage']],
            ['Network', infra_recommendations['network']],
            ['Load Balancer', infra_recommendations['load_balancer']],
            ['Container Orchestration', infra_recommendations['scaling']],
            ['Application Replicas', infra_recommendations['replicas']],
            ['Database', infra_recommendations['database']],
            ['Caching', infra_recommendations['caching']]
        ]

        if 'cdn' in infra_recommendations:
            infra_table_data.append(['CDN', infra_recommendations['cdn']])
        if 'monitoring' in infra_recommendations:
            infra_table_data.append(['Monitoring', infra_recommendations['monitoring']])
        if 'backup' in infra_recommendations:
            infra_table_data.append(['Backup Strategy', infra_recommendations['backup']])

        infra_table = Table(infra_table_data, colWidths=[2*inch, 4*inch])
        infra_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))

        story.append(infra_table)
        story.append(Spacer(1, 20))

        footer = Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data Cleanup Micro-SaaS v2.0",
            self.styles['Normal']
        )
        story.append(Spacer(1, 30))
        story.append(footer)

        doc.build(story)
        return output_path
